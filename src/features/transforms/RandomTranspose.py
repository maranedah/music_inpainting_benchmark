import torch
class RandomTranspose:
    def __init__(self, bounds, representation, is_train):
        self.bounds = bounds
        self.representation = representation
        self.is_train = is_train


    def __call__(self, data):
        if self.is_train:
            return data
        
        min_value = self.bounds[0]
        max_value = self.bounds[1]
            
        if self.representation == "noteseq":
            data = torch.from_numpy(data)
            pitch_classes = data[0][data[0]<128]
            upper_bound = min(6, max_value - pitch_classes.max())
            lower_bound =  max(-6, min_value - pitch_classes.min())
            i = torch.randint(low=lower_bound, high=upper_bound+1, size=(1,))
            d_plus = torch.where(data[0] < 128, 1, 0) * i
            d_plus = d_plus.unsqueeze(0)
            r = data + d_plus
            r = r.cpu().detach().numpy()

        if self.representation == "remi":
            data = torch.Tensor(data)
            data = data.type(torch.int)
            pitch_classes = data[:,3]
            upper_bound = min(6, max_value - pitch_classes.max())
            lower_bound =  max(-6, min_value - pitch_classes.min())
            i = torch.randint(low=lower_bound, high=upper_bound+1, size=(1,))
        
            data[:,3] = data[:,3] + i
            r = data.tolist()
        return r