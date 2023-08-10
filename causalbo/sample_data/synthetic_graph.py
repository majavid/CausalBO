from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch

class SyntheticGraph(object):
    def age(num_data_points):
        return (55 - 75) * torch.rand(num_data_points, 1) + 75
    
    def bmi(input_tensor):
        input_tensor = input_tensor[..., :1]
        return torch.normal((torch.full_like(input_tensor, 27.) - 0.01 * input_tensor), 0.7)

    def aspirin(input_tensor):
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[-8 + 0.1 * i[0] + 0.03 * i[1]] for i in input_tensor])
        return torch.nn.Sigmoid()(new_tensor)

    def statin(input_tensor):
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[-13 + 0.1 * i[0] + 0.2 * i[1]] for i in input_tensor])
        return torch.nn.Sigmoid()(new_tensor)
    
    def cancer(input_tensor):
        input_tensor = input_tensor[..., :4]
        new_tensor = torch.tensor([[2.2 - 0.05 * i[0] + 0.01 * i[1] - 0.04 * i[2] + 0.02 * i[3]] for i in input_tensor])
        return torch.nn.Sigmoid()(new_tensor)
    
    def psa(input_tensor):
        

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

    def draw(self):
        self.graph.draw()

    

        

    

