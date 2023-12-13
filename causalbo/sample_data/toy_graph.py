from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch

# Sample DAG and SCM using toy dataset provided by V. Aglietti et al.
# CausalBO does not require data to be organized in this fashion, but it does help to keep it organized in a similar manner.
class ToyGraph(object):
    # epsilon_X
    def X(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return input_tensor + noise

    # exp(−X) + epsilon_Z
    def Z(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return (torch.exp(-input_tensor)) + noise

    # cos(Z) − exp(−Z/20) + epsilon_Y
    def Y(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return ((torch.cos(input_tensor)) - (torch.exp(-input_tensor / 20))) + noise
    
    def __init__(self, num_observations = 1000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        # Interventional domain
        self.interventional_domain = {'X': [-5,5], 'Z': [-5,20]}

        # Graph structure
        self.graph = SCM([('X', 'Z'), ('Z', 'Y')])

        # Same structure, deep copy
        self.true_graph = SCM([('X', 'Z'), ('Z', 'Y')])

        # Generate observational data
        obs_data_x = ToyGraph.X(torch.linspace(-5, 5, num_observations).view(-1,1), noise_stdev=1)
        obs_data_z = ToyGraph.Z(obs_data_x, noise_stdev=1)
        obs_data_y = ToyGraph.Y(obs_data_z, noise_stdev=1)

        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['X'] = torch.flatten(obs_data_x).tolist()
        self.observational_samples['Z'] = torch.flatten(obs_data_z).tolist()
        self.observational_samples['Y'] = torch.flatten(obs_data_y).tolist()
        # Shuffle dataframe into random order
        self.observational_samples.sample(frac=1)
        # Fit graph to observational data.
        self.graph.fit(self.observational_samples)

        # Generate objective data
        obs_data_x = ToyGraph.X(torch.linspace(-5, 5, num_objective_points).view(-1,1))
        obs_data_z = ToyGraph.Z(obs_data_x)
        obs_data_y = ToyGraph.Y(obs_data_z)

        # Add to dataframe
        self.objective_samples = DataFrame()
        self.objective_samples['X'] = torch.flatten(obs_data_x).tolist()
        self.objective_samples['Z'] = torch.flatten(obs_data_z).tolist()
        self.objective_samples['Y'] = torch.flatten(obs_data_y).tolist()

        # Fit graph to objective data.
        self.true_graph.fit(self.objective_samples)        

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

