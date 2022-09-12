import random

import torch
from torch import distributions, nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import Parameter
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        note_embedding_dim,
        rnn_hidden_size,
        num_layers,
        num_notes,
        dropout,
        bidirectional,
        z_dim,
        rnn_class,
    ):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.note_embedding_dim = note_embedding_dim
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.z_dim = z_dim
        self.dropout = dropout
        self.rnn_class = rnn_class
        self.lstm = self.rnn_class(
            input_size=note_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.num_notes = num_notes
        self.note_embedding_layer = nn.Embedding(
            self.num_notes, self.note_embedding_dim
        )

        self.linear_mean = nn.Sequential(
            nn.Linear(
                self.rnn_hidden_size * self.num_directions * self.num_layers,
                self.rnn_hidden_size * self.num_directions,
            ),
            nn.SELU(),
            nn.Linear(self.rnn_hidden_size * self.num_directions, z_dim),
        )

        self.linear_log_std = nn.Sequential(
            nn.Linear(
                self.rnn_hidden_size * self.num_directions * self.num_layers,
                self.rnn_hidden_size * self.num_directions,
            ),
            nn.SELU(),
            nn.Linear(self.rnn_hidden_size * self.num_directions, z_dim),
        )

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return (
            f"Encoder("
            f"{self.note_embedding_dim},"
            f"{self.rnn_class},"
            f"{self.num_layers},"
            f"{self.rnn_hidden_size},"
            f"{self.dropout},"
            f"{self.bidirectional},"
            f"{self.z_dim},"
            f")"
        )

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def hidden_init(self, batch_size):
        """
        Initializes the hidden state of the encoder GRU
        :param batch_size: int
        :return: torch tensor,
        (
            self.num_encoder_layers x self.num_directions,
            batch_size,
            self.encoder_hidden_size
        )
        """
        hidden = torch.zeros(
            (self.num_layers * self.num_directions, batch_size, self.rnn_hidden_size)
        ).to(torch.cuda.current_device())
        return hidden

    def embed_forward(self, score_tensor):
        """
        Performs the forward pass of the embedding layer
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :return: torch tensor,
                (batch_size, measure_seq_len, embedding_size)
        """
        x = self.note_embedding_layer(score_tensor)
        return x

    def forward(self, score_tensor):
        """
        Performs the forward pass of the model, overrides torch method
        :param score_tensor: torch Variable
                (batch_size, measure_seq_len)
        :return: torch distribution
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                nan_check = torch.isnan(param.data)
                if nan_check.nonzero().size(0) > 0:
                    print("Encoder has become nan")
                    raise ValueError

        batch_size, measure_seq_len = score_tensor.size()

        # embed score
        embedded_seq = self.embed_forward(score_tensor=score_tensor)

        # pass through RNN
        hidden = self.hidden_init(batch_size)
        _, hidden = self.lstm(embedded_seq, hidden)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(batch_size, -1)

        # compute distribution parameters
        z_mean = self.linear_mean(hidden)
        z_log_std = self.linear_log_std(hidden)

        # print(z_mean, z_log_std)
        z_distribution = Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution


class Decoder(nn.Module):
    def __init__(
        self,
        note_embedding_dim,
        num_notes,
        z_dim,
    ):
        super(Decoder, self).__init__()
        self.name = "DecoderABC"
        self.num_notes = num_notes
        self.note_embedding_dim = note_embedding_dim
        self.z_dim = z_dim
        self.note_embedding_layer = nn.Embedding(
            self.num_notes, self.note_embedding_dim
        )

    def forward(self, z, score_tensor, train=None):
        """
        :param z: torch_tensor, latent variable
        :param score_tensor: torch_tensor, original measure score tensor
        :param train: bool
        :return:
        """
        return None, None

    def check_index(self, indices):
        """
        :param indices: int,
        :return: bool
        """
        indices = indices.cpu()
        if min(indices) >= 0 and max(indices) < self.num_notes:
            return True
        else:
            print("Invalid Values of Indices: ", min(indices), max(indices))
            raise ValueError

    def xavier_initialization(self):
        """
        Initializes the network params
        :return:
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)


class SRDecoder(Decoder):
    def __init__(
        self,
        note_embedding_dim,
        num_notes,
        z_dim,
        num_layers,
        rnn_hidden_size,
        dropout,
        rnn_class,
    ):
        super(SRDecoder, self).__init__(note_embedding_dim, num_notes, z_dim)
        self.name = "SRDecoder"
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        # define the individual layers
        self.z_to_rnn_input = nn.Sequential(
            nn.Linear(self.z_dim, self.rnn_hidden_size),
            nn.SELU(),
            nn.Linear(self.rnn_hidden_size, self.note_embedding_dim),
        )

        self.x_0 = Parameter(data=torch.zeros(note_embedding_dim))
        self.rnn_class = rnn_class
        self.rnn_dec = self.rnn_class(
            input_size=2 * self.note_embedding_dim,  # input to bear RNN will be zeros
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.rnn_out_to_note_emb = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.num_notes), nn.ReLU()
        )

        self.use_teacher_forcing = True
        self.teacher_forcing_prob = 0.5
        self.sampling = "argmax"
        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return (
            f"{self.name}"
            f"{self.note_embedding_dim},"
            f"{self.rnn_class},"
            f"{self.num_layers},"
            f"{self.rnn_hidden_size},"
            f"{self.dropout},"
            f")"
        )

    def hidden_init(self, batch_size):
        """
        :param batch_size: int,
        :return: torch tensor,
                (self.num_layers, batch_size, self.rnn_hidden_size)
        """
        h = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size)
        return h

    def forward(self, z, score_tensor, train):
        """
        :param z: torch tensor,
                (batch_size, self.z_dim)
        :param score_tensor: torch tensor
                (batch_size, measure_seq_len)
        :return: weights: torch tensor,
                (batch_size, measure_seq_len, self.num_notes)
                samples: torch tensor,
                (batch_size, measure_seq_len)
        """
        if self.use_teacher_forcing and train:
            teacher_forced = random.random() < self.teacher_forcing_prob
        else:
            teacher_forced = False
        if not train:
            sampling = "argmax"
        else:
            sampling = self.sampling

        batch_size_z, z_dim = z.size()
        assert z_dim == self.z_dim
        batch_size, measure_seq_len = score_tensor.size()
        assert batch_size == batch_size_z

        # compute output of rnn_dec
        weights, samples = self.forward_rnn(z, score_tensor, teacher_forced, sampling)
        return weights, samples

    def forward_rnn(self, z, score_tensor, teacher_forced, sampling):
        """
        :param z: torch tensor,
                (batch_size, self.z_dim):
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :param teacher_forced: bool,
        :param sampling: string
        :return:
        """
        samples = []
        weights = []
        batch_size, measure_seq_len = score_tensor.size()
        hidden = self.hidden_init(batch_size)
        rnn_input = self.x_0.unsqueeze(0).expand(batch_size, self.note_embedding_dim)
        rnn_input = rnn_input.unsqueeze(1)
        rnn_input_emb = self.z_to_rnn_input(z)
        rnn_input_emb = rnn_input_emb.unsqueeze(1)
        for i in range(measure_seq_len):
            rnn_input = torch.cat((rnn_input, rnn_input_emb), 2)
            rnn_out, hidden = self.rnn_dec(rnn_input, hidden)
            probs = self.rnn_out_to_note_emb(rnn_out[:, 0, :])
            # sample and embed next rnn_input
            if self.use_teacher_forcing and teacher_forced:
                indices = score_tensor.detach()[:, i]
                indices = indices.unsqueeze(1)
                assert self.check_index(indices)
            else:
                if sampling == "multinomial":
                    softmax = F.softmax(probs.detach(), dim=1)
                    indices = torch.multinomial(softmax, 1)
                    assert self.check_index(indices)
                elif sampling == "argmax":
                    _, indices = probs.detach().topk(k=1, dim=1)
                    try:
                        self.check_index(indices)
                    except ValueError:
                        print(probs)
                        raise ValueError
                else:
                    raise NotImplementedError
            rnn_input = self.note_embedding_layer(indices)
            samples.append(indices[:, :, None])
            # save all
            probs = probs.view(batch_size, self.num_notes)
            weights.append(probs[:, None, :])
        weights = torch.cat(weights, 1)
        samples = torch.cat(samples, 2)
        return weights, samples


class SRDecoderNoInput(SRDecoder):
    def __init__(
        self,
        note_embedding_dim,
        num_notes,
        z_dim,
        num_layers,
        rnn_hidden_size,
        dropout,
        rnn_class,
    ):
        super(SRDecoderNoInput, self).__init__(
            note_embedding_dim,
            num_notes,
            z_dim,
            num_layers,
            rnn_hidden_size,
            dropout,
            rnn_class,
        )
        self.name = "SRDecoderNoInput"
        # define the individual layers
        self.z_to_rnn_input = nn.Sequential(
            nn.Linear(self.z_dim, self.rnn_hidden_size),
        )

        self.x_0 = Parameter(data=torch.zeros(note_embedding_dim))
        self.rnn_class = rnn_class
        self.rnn_dec = self.rnn_class(
            input_size=self.rnn_hidden_size,  # input to bear RNN will be zeros
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.rnn_out_to_note_emb = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.num_notes), nn.ReLU()
        )

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return (
            f"{self.name}"
            f"{self.note_embedding_dim},"
            f"{self.rnn_class},"
            f"{self.num_layers},"
            f"{self.rnn_hidden_size},"
            f"{self.dropout},"
            f")"
        )

    def hidden_init(self, batch_size):
        """
        :param batch_size: int,
        :return: torch tensor,
                (self.num_layers, batch_size, self.rnn_hidden_size)
        """
        h = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size)
        return h

    def forward_rnn(self, z, score_tensor, teacher_forced, sampling):
        """
        :param z: torch tensor,
                (batch_size, self.z_dim):
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :param teacher_forced: bool,
        :param sampling: string
        :return:
        """
        batch_size, measure_seq_len = score_tensor.size()
        hidden = self.hidden_init(batch_size=batch_size)
        rnn_input = self.z_to_rnn_input(z)
        rnn_input = rnn_input.unsqueeze(1).expand(-1, measure_seq_len, -1)
        rnn_out, hidden = self.rnn_dec(rnn_input, hidden)
        rnn_out = rnn_out.contiguous().view(batch_size * measure_seq_len, -1)
        weights = self.rnn_out_to_note_emb(rnn_out)
        weights = weights.contiguous().view(batch_size, measure_seq_len, -1)
        if sampling == "multinomial":
            softmax = F.softmax(weights.detach, dim=2)
            samples = torch.multinomial(softmax, 2)
        elif sampling == "argmax":
            _, samples = weights.detach().topk(k=1, dim=2)
        return weights, samples


class HierarchicalDecoder(Decoder):
    def __init__(
        self,
        note_embedding_dim,
        num_notes,
        z_dim,
        num_layers,
        rnn_hidden_size,
        dropout,
        rnn_class,
    ):
        super(HierarchicalDecoder, self).__init__(note_embedding_dim, num_notes, z_dim)
        self.name = "HierarchicalDecoder"
        self.rnn_class = rnn_class
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        # define the individual layers
        self.z_to_beat_rnn_input = nn.Sequential(
            nn.Linear(self.z_dim, self.rnn_hidden_size * self.num_layers), nn.SELU()
        )

        self.beat_rnn_input_dim = 1
        self.b_0 = Parameter(data=torch.zeros(self.beat_rnn_input_dim))
        self.rnn_beat = self.rnn_class(
            input_size=self.beat_rnn_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.beat_emb_to_tick_rnn_hidden = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size * self.num_layers),
            nn.SELU(),
        )

        self.beat_emb_to_tick_rnn_input = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size), nn.SELU()
        )

        self.x_0 = Parameter(data=torch.zeros(note_embedding_dim))
        self.rnn_tick = self.rnn_class(
            input_size=self.note_embedding_dim
            + self.rnn_hidden_size,  # input to bear RNN will be zeros
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.tick_emb_to_note_emb = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.num_notes), nn.ReLU()
        )

        self.use_teacher_forcing = True
        self.teacher_forcing_prob = 0.5
        self.sampling = "argmax"
        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return (
            f"{self.name}"
            f"{self.note_embedding_dim},"
            f"{self.rnn_class},"
            f"{self.num_layers},"
            f"{self.rnn_hidden_size},"
            f"{self.dropout},"
            f")"
        )

    def hidden_init(self, inp, rnn_type):
        """
        Initializes the hidden state based on the RNN type
        :param inp: torch tensor,
                (batch_size, num_feats)
        :param rnn_type: str, 'beat' for beat rnn, 'tick' for tick rnn
        :return: torch tensor,
                (self.num_layers, batch_size, self.rnn_hidden_size)
        """
        batch_size = inp.size(0)
        if rnn_type == "beat":
            h_0 = self.z_to_beat_rnn_input(inp)
        elif rnn_type == "tick":
            h_0 = self.beat_emb_to_tick_rnn_hidden(inp)
        else:
            raise ValueError
        h = h_0.view(batch_size, self.num_layers, -1)
        h = h.transpose(0, 1).contiguous()
        return h

    def forward(self, z, score_tensor, train):
        """
        Performs the forward pass of the model, overrides torch method
        :param z: torch tensor,
                (batch_size, self.z_dim)
        :param score_tensor: torch tensor
                (batch_size, measure_seq_len)
        :return: weights: torch tensor,
                (batch_size, measure_seq_len, self.num_notes)
                samples: torch tensor,
                (batch_size, measure_seq_len)
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                nan_check = torch.isnan(param.data)
                if nan_check.nonzero().size(0) > 0:
                    print("Decoder has become nan")
                    raise ValueError

        if self.use_teacher_forcing and train:
            teacher_forced = random.random() < self.teacher_forcing_prob
        else:
            teacher_forced = False
        if not train:
            sampling = "argmax"
        else:
            sampling = self.sampling

        batch_size_z, z_dim = z.size()
        assert z_dim == self.z_dim
        batch_size, measure_seq_len = score_tensor.size()
        assert batch_size == batch_size_z

        # compute output of beat rnn
        beat_seq_len = 4
        beat_rnn_out = self.forward_beat_rnn(z, beat_seq_len)

        # compute output of tick rnn
        tick_seq_len = 6
        weights, samples = self.forward_tick_rnn(
            score_tensor, beat_rnn_out, tick_seq_len, teacher_forced, sampling
        )

        return weights, samples

    def forward_beat_rnn(self, z, seq_len):
        """
        Computes the forward pass of the Beat RNN
        :param z: torch tensor,
                (batch_size, self.z_dim)
        :param seq_len: int, sequence length for beat RNN unrolling
        :return:
        """
        batch_size = z.size(0)
        hidden = self.hidden_init(z, rnn_type="beat")
        beat_rnn_input = self.b_0.unsqueeze(0).expand(
            batch_size, seq_len, self.beat_rnn_input_dim
        )
        beat_rnn_out, _ = self.rnn_beat(beat_rnn_input, hidden)
        return beat_rnn_out

    def forward_tick_rnn(
        self, score_tensor, beat_rnn_out, tick_seq_len, teacher_forced, sampling
    ):
        """
        Computes the forward pass of the Tick RNN
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :param beat_rnn_out: torch tensor,
                (batch_size, beat_seq_len, self.rnn_hidden_size)
        :param tick_seq_len: int, sequence length for tick RNN unrolling
        :param teacher_forced: bool, whether to use teacher forcing or not
        :param sampling: string, which sampling method to use
        :return:
        """
        samples = []
        weights = []
        batch_size, beat_seq_len, _ = beat_rnn_out.size()
        tick_rnn_input = self.x_0.unsqueeze(0).expand(
            batch_size, self.note_embedding_dim
        )
        tick_rnn_input = tick_rnn_input.unsqueeze(1)
        for i in range(beat_seq_len):
            hidden = self.hidden_init(beat_rnn_out[:, i, :], rnn_type="tick")
            beat_emb_input = self.beat_emb_to_tick_rnn_input(
                beat_rnn_out[:, i, :]
            ).unsqueeze(1)
            for j in range(tick_seq_len):
                tick_rnn_input = torch.cat((tick_rnn_input, beat_emb_input), 2)
                tick_rnn_out, hidden = self.rnn_tick(tick_rnn_input, hidden)
                probs = self.tick_emb_to_note_emb(tick_rnn_out[:, 0, :])
                # sample and embed next rnn_input
                if self.use_teacher_forcing and teacher_forced:
                    indices = score_tensor.detach()[:, i * tick_seq_len + j]
                    indices = indices.unsqueeze(1)
                    assert self.check_index(indices)
                else:
                    if sampling == "multinomial":
                        softmax = F.softmax(probs.detach(), dim=1)
                        indices = torch.multinomial(softmax, 1)
                        assert self.check_index(indices)
                    elif sampling == "argmax":
                        _, indices = probs.detach().topk(k=1, dim=1)
                        try:
                            self.check_index(indices)
                        except ValueError:
                            print(probs)
                            raise ValueError
                    else:
                        raise NotImplementedError
                tick_rnn_input = self.note_embedding_layer(indices)
                samples.append(indices[:, :, None])
                # save all
                probs = probs.view(batch_size, self.num_notes)
                weights.append(probs[:, None, :])
        weights = torch.cat(weights, 1)
        samples = torch.cat(samples, 2)
        return weights, samples


class MeasureVAE(nn.Module):
    def __init__(
        self,
        note_embedding_dim=10,
        metadata_embedding_dim=2,
        num_encoder_layers=2,
        encoder_hidden_size=512,
        encoder_dropout_prob=0.5,
        latent_space_dim=256,
        num_decoder_layers=2,
        decoder_hidden_size=512,
        decoder_dropout_prob=0.5,
        has_metadata=False,
    ):
        """
        Initializes the Measure VAE class object
        :param dataset: FolkMeasureDataset object
        :param note_embedding_dim: int,
        :param metadata_embedding_dim: int,
        :param num_encoder_layers: int,
        :param encoder_hidden_size: int,
        :param encoder_dropout_prob: float, from 0. to 1.
        :param latent_space_dim: int,
        :param num_decoder_layers: int,
        :param decoder_hidden_size: int,
        :param decoder_dropout_prob: float, from 0. to 1.
        :param has_metadata: bool
        """
        super(MeasureVAE, self).__init__()
        # define constants
        self.num_beats_per_measure = 4
        self.num_ticks_per_measure = 24
        self.num_ticks_per_beat = int(
            self.num_ticks_per_measure / self.num_beats_per_measure
        )

        # initialize members
        # self.dataset = dataset.__repr__()
        self.note_embedding_dim = note_embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout_prob = encoder_dropout_prob
        self.latent_space_dim = latent_space_dim
        self.num_decoder_layers = num_decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_dropout_prob = decoder_dropout_prob
        self.has_metadata = has_metadata
        self.num_notes = 130  # len(dataset.note2index_dicts[0])
        print("NUMBER OF NOTES: ", self.num_notes)
        # Encoder
        self.encoder = Encoder(
            note_embedding_dim=self.note_embedding_dim,
            rnn_hidden_size=self.encoder_hidden_size,
            num_layers=self.num_encoder_layers,
            num_notes=self.num_notes,
            dropout=self.encoder_dropout_prob,
            bidirectional=True,
            z_dim=self.latent_space_dim,
            rnn_class=torch.nn.GRU,
        ).to(torch.cuda.current_device())

        # Decoder
        self.decoder = HierarchicalDecoder(
            note_embedding_dim=self.note_embedding_dim,
            num_notes=self.num_notes,
            z_dim=self.latent_space_dim,
            num_layers=self.num_decoder_layers,
            rnn_hidden_size=self.decoder_hidden_size,
            dropout=self.decoder_dropout_prob,
            rnn_class=torch.nn.GRU,
        ).to(torch.cuda.current_device())

    def forward(self, measure_score_tensor: Variable, train=True):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :param measure_metadata_tensor: torch Variable,
                (batch_size, measure_seq_length, num_metadata)
        :param train: bool,
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        seq_len = measure_score_tensor.size(1)
        assert seq_len == self.num_ticks_per_measure
        # compute output of encoding layer
        z_dist = self.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = distributions.Normal(
            loc=torch.zeros_like(z_dist.loc), scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # compute output of decoding layer
        weights, samples = self.decoder(
            z=z_tilde, score_tensor=measure_score_tensor, train=train
        )
        return weights, samples, z_dist, prior_dist, z_tilde, z_prior
