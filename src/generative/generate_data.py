import src.models as models
import torch 
def generate_data(model, data, model_name):
    if model_name == "VLI":
        y_pred = models.variable_length_piano_infilling.train_model.predict(model, data.clone())
        #y_pred = torch.from_numpy(y_pred)
        
        inpaint_mask = torch.where(6 <= data[:,1], 1, 0) + torch.where(data[:,1] <= 9, 1, 0) - 1
        inpaint_indexes = torch.nonzero(inpaint_mask).flatten()
        y_true = torch.index_select(data, 0, inpaint_indexes).unsqueeze(0)

        past_mask = torch.where(data[:,1] <= 5, 1, 0)
        past_indexes = torch.nonzero(past_mask).flatten()
        y_past = torch.index_select(data, 0, past_indexes).unsqueeze(0)

        future_mask = torch.where(data[:,1] >= 10, 1, 0) + torch.where(data[:,1] <= 15, 1, 0) - 1
        future_indexes = torch.nonzero(future_mask).flatten()
        y_future = torch.index_select(data, 0, future_indexes).unsqueeze(0)

        y_true = y_true.reshape(-1, y_true.shape[-1]).detach().to('cpu').numpy()
        y_past = y_past.reshape(-1, y_past.shape[-1]).detach().to('cpu').numpy()
        y_future = y_future.reshape(-1, y_future.shape[-1]).detach().to('cpu').numpy()

    if model_name == "ARNN":
        y_pred = models.anticipation_rnn.train_model.generate(model,data).detach().to('cpu').numpy()
        y_true = data[0][24*6:24*10] 
        y_past = data[0][:24*6] 
        y_future = data[0][24*10:] 

    if model_name == "INPAINTNET":
        y_pred = models.inpaintnet.train_model.generate(model,data).detach().to('cpu').numpy()
        y_true = data[24*6:24*10] 
        y_past = data[:24*6] 
        y_future = data[24*10:] 

    if model_name == "SKETCHNET":
        
        y_pred = models.music_sketchnet.train_model.generate(model,data).detach().to('cpu').numpy()
        y_past = data['past_x'][4].flatten().detach().to('cpu').numpy()
        y_future = data['future_x'][4].flatten().detach().to('cpu').numpy()
        y_true = data['middle_x'][4].flatten().detach().to('cpu').numpy()

    if model_name == "GRU_VAE":
        
        y_pred = models.gru_vae.train_model.generate(model,data).detach().to('cpu').numpy()
        y_past = data['past_x'][4].flatten().detach().to('cpu').numpy()
        y_future = data['future_x'][4].flatten().detach().to('cpu').numpy()
        y_true = data['middle_x'][4].flatten().detach().to('cpu').numpy()

    return y_pred, y_true, y_past, y_future