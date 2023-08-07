from pandas import DataFrame
from do_calculus import SCM
import torch

class ToyGraph(object):
    def X(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return input_tensor + noise

    def Z(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return (torch.exp(-input_tensor)) + noise

    def Y(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return ((torch.cos(input_tensor)) - (torch.exp(-input_tensor / 20))) + noise
    # def X(input, noise_mean=0, noise_stdev=0):
    #     return input + np.random.normal(noise_mean, noise_stdev)

    # def Z(input, noise_mean=0, noise_stdev=0):
    #     return (math.e ** -input) + np.random.normal(noise_mean, noise_stdev)

    # def Y(input, noise_mean=0, noise_stdev=0):  
    #     return ((math.cos(input)) - (math.e ** (-input / 20))) + np.random.normal(noise_mean, noise_stdev)
    
    def __init__(self):
        # Interventional domain
        self.interventional_domain = {'X': [-5,5], 'Z': [-5,20]}
        # Graph structure
        self.graph = SCM([('X', 'Z'), ('Z', 'Y')])
        # Observational data samples
        self.observational_samples = DataFrame()
        # Objective function
        self.obj_func = {'X': lambda x: ToyGraph.Y(ToyGraph.Z(ToyGraph.X(x))),
                         'Z': lambda z: ToyGraph.Y(ToyGraph.Z(z)),
                         'Y': lambda y: ToyGraph.Y(y)}
        # Generate observational data
        #obs_data_x = [ToyGraph.X(np.random.normal(0, 1), noise_stdev=1) for x in range(0,1000)]
        obs_data_x = ToyGraph.X(torch.randn(1000, 1).view(-1,1), noise_stdev=1)
        #obs_data_z = [ToyGraph.Z(x, noise_stdev=1) for x in obs_data_x]
        obs_data_z = ToyGraph.Z(obs_data_x, noise_stdev=1)
        #obs_data_y = [ToyGraph.Y(z, noise_stdev=1) for z in obs_data_z]
        obs_data_y = ToyGraph.Y(obs_data_z, noise_stdev=1)
        # Add to dataframe
        self.observational_samples['X'] = torch.flatten(obs_data_x).tolist()
        self.observational_samples['Z'] = torch.flatten(obs_data_z).tolist()
        self.observational_samples['Y'] = torch.flatten(obs_data_y).tolist()

    

        

    

