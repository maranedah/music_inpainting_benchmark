import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CyclicLR


# Definir la red
class SketchNet_V2(nn.Module):
    def __init__(self, zp_dims, zr_dims, pf_num, inpaint_len, gen_dims, vae_model):
        super(SketchNet_V2, self).__init__()
        self.vae_model = vae_model

        self.zp_dims = zp_dims
        self.zr_dims = zr_dims
        self.pf_num = pf_num
        self.inpaint_len = inpaint_len
        self.gen_dims = gen_dims

        # Capa LSTM
        self.p_gru = nn.GRU(
            zp_dims * 2,
            gen_dims,
            pf_num,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )
        self.r_gru = nn.GRU(
            zr_dims * 2,
            gen_dims,
            pf_num,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.gen_p_gru = nn.GRU(
            zp_dims, gen_dims, pf_num, batch_first=True, bidirectional=True, dropout=0.5
        )
        self.gen_r_gru = nn.GRU(
            zr_dims, gen_dims, pf_num, batch_first=True, bidirectional=True, dropout=0.5
        )

        # Generation Output
        self.gen_p_out = nn.Linear(gen_dims * 2, zp_dims)
        self.gen_r_out = nn.Linear(gen_dims * 2, zr_dims)

        self.xavier_initialization()

    def gen_pitch_decoder(self, pf_pz, past_z, teacher_ratio, middle_z):
        y = past_z[:, -1, : self.zp_dims].unsqueeze(1)  # Get last measure in Z past
        hxx = pf_pz
        z_pitches = []

        for i in range(self.inpaint_len):
            y, hxx = self.gen_p_gru(y, hxx)
            y = y.contiguous().view(y.size(0), -1)
            y = self.gen_p_out(y)
            z_pitches.append(y)
            # Teacher forcing
            if self.training and torch.rand(1).item() < teacher_ratio:
                y = middle_z[:, i, : self.zp_dims]
            y = y.unsqueeze(1)

        return torch.stack(z_pitches, 1)

    def gen_rhythm_decoder(self, pf_z, past_z, teacher_ratio, middle_z):
        y = past_z[:, -1, self.zp_dims :].unsqueeze(1)  # Get last measure in Z
        hxx = pf_z
        z_rhythms = []
        for i in range(self.inpaint_len):
            y, hxx = self.gen_r_gru(y, hxx)
            y = y.contiguous().view(y.size(0), -1)
            y = self.gen_r_out(y)
            z_rhythms.append(y)
            # Teacher forcing
            if self.training and torch.rand(1).item() < teacher_ratio:
                y = middle_z[:, i, self.zp_dims :]
            y = y.unsqueeze(1)
        return torch.stack(z_rhythms, 1)

    def get_pf_pz(self, past_pz, future_pz):
        # Input: [64, 6, 128], [64, 6, 128] (batch, n_measures, Z_vector)
        # Output: [4, 64, 256?] -> Es la dimensión de la hidden del LSTM principal
        pz = torch.cat((past_pz, future_pz), 2)
        _, h = self.p_gru(pz)
        pf_pz = h
        return pf_pz

    # Concat Past and Future "Embedding" for rhythm
    def get_pf_rz(self, past_rz, future_rz):
        # Input: [64, 6, 128], [64, 6, 128] (batch, n_measures, Z_vector)
        # Output: [4, 64, 256?] -> Es la dimensión de la hidden del LSTM principal
        rz = torch.cat((past_rz, future_rz), 2)
        _, h = self.r_gru(rz)
        pf_rz = h
        return pf_rz

    def forward(self, past_x, future_x, middle_x):
        past_z = self.get_z_seq(past_x)
        future_z = self.get_z_seq(future_x)
        middle_z = self.get_z_seq(middle_x)

        past_pz = past_z[:, :, : self.zp_dims]
        past_rz = past_z[:, :, self.zp_dims :]
        future_pz = future_z[:, :, : self.zp_dims]
        future_rz = future_z[:, :, self.zp_dims :]

        pf_pz = self.get_pf_pz(past_pz, future_pz)
        pf_rz = self.get_pf_rz(past_rz, future_rz)

        # Teacher Forcing
        teacher_ratio = 0.5

        # Autoreggresive generation of new Z measures
        gen_pz = self.gen_pitch_decoder(pf_pz, past_z, teacher_ratio, middle_z)
        gen_rz = self.gen_rhythm_decoder(pf_rz, past_z, teacher_ratio, middle_z)
        gen_z = torch.cat((gen_pz, gen_rz), -1)
        gen_m = self.get_measure(gen_z)
        return gen_m

    def get_measure(self, z):
        dummy = torch.zeros((z.size(0), self.vae_model.seq_len)).long().cuda()
        ms = []
        for i in range(self.inpaint_len):
            m = self.vae_model.final_decoder(z[:, i, :], dummy, is_train=False)
            ms.append(m)
        return torch.stack(ms, 1)

    def get_z_seq(self, x):
        if x is None:
            return None
        # order: px, rx, len_x, nrx, gd
        px, _, len_x, nrx, gd = x
        batch_size = px.size(0)
        px = px.view(-1, self.vae_model.seq_len)
        nrx = nrx.view(-1, self.vae_model.seq_len, 3)
        len_x = len_x.view(-1)
        p_dis = self.vae_model.pitch_encoder(px, len_x)
        r_dis = self.vae_model.rhythm_encoder(nrx)
        zp = p_dis.rsample()
        zr = r_dis.rsample()
        z = torch.cat((zp, zr), -1)
        z = z.view(batch_size, -1, self.zr_dims + self.zp_dims)
        return z

    def inpaint(self, past, future):
        dummy = None
        logits = self(past, future, dummy)
        y_pred = logits.view(-1, logits.size(-1)).max(-1)[-1]
        return y_pred

    def xavier_initialization(self):
        layers = self._modules.keys()
        for layer in layers:
            if layer != "vae_model":
                for name, param in self._modules[layer].named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)


class LitSketchNet_V2(SketchNet_V2, pl.LightningModule):
    def __init__(
        self,
        zp_dims,
        zr_dims,
        pf_num,
        inpaint_len,
        gen_dims,
        vae_model,
        learn_rate: float,
        batch_size: int,
        sch_factor: float,
        sch_step: int,
        sch_mode: str,
        sch_gamma: float,
        sch_momentum: bool,
    ) -> None:
        super(SketchNet_V2, self).__init__()
        super(LitSketchNet_V2, self).__init__(
            zp_dims, zr_dims, pf_num, inpaint_len, gen_dims, vae_model
        )
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.sch_factor = sch_factor
        self.sch_step = sch_step
        self.sch_mode = sch_mode
        self.sch_gamma = sch_gamma
        self.sch_momentum = sch_momentum
        self.save_hyperparameters()

        self.loss_fn = F.cross_entropy

    def training_step(self, batch, batch_idx):
        d, _ = batch
        label = d["inpaint_gd_whole"]
        recon_x = self(d["past_x"], d["future_x"], d["middle_x"])
        loss = self.loss_fn(
            recon_x.view(-1, recon_x.size(-1)), label.view(-1), reduction="mean"
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        d, _ = batch
        label = d["inpaint_gd_whole"]
        recon_x = self(d["past_x"], d["future_x"], d["middle_x"])
        return recon_x.view(-1, recon_x.size(-1)), label.view(-1)

    def validation_epoch_end(self, val_step_outputs):
        """
        Computes losses or metrics on validation dataloader
        when epoch ends.
        val_step_outputs: list of validation_step outputs during the epoch
        """
        model_out, labels = zip(*val_step_outputs)
        model_out = torch.cat(model_out)
        labels = torch.cat(labels)
        val_loss = self.loss_fn(model_out, labels)
        self.log("val_loss", val_loss)

        # Validation metrics
        # ...

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
        print(type(self.learn_rate), type(self.sch_factor))
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.learn_rate,
            max_lr=self.learn_rate * self.sch_factor,
            step_size_up=self.sch_step,
            mode=self.sch_mode,
            gamma=self.sch_gamma,
            cycle_momentum=self.sch_momentum,
        )
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler_config": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val_loss",
                },
            },
        )
