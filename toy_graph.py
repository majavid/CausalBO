import numpy as np
import math
import pandas as pd
import do_calculus

class ToyGraph(object):
    def X(input, noise_mean=0, noise_stdev=0):
        return input + np.random.normal(noise_mean, noise_stdev)

    def Z(input, noise_mean=0, noise_stdev=0):
        return (math.e ** -input) + np.random.normal(noise_mean, noise_stdev)

    def Y(input, noise_mean=0, noise_stdev=0):  
        return ((math.cos(input)) - (math.e ** (-input / 20))) + np.random.normal(noise_mean, noise_stdev)
    
    def __init__(self):
        # Interventional domain
        self.interventional_domain = {'X': [-5,5], 'Z': [-5,20]}
        # Graph structure
        self.graph = do_calculus.SCM([('X', 'Z'), ('Z', 'Y')])
        # Observational data samples
        self.observational_samples = pd.DataFrame()
        # Objective function
        self.obj_func = lambda x: ToyGraph.Y(ToyGraph.Z(ToyGraph.X(x)))
        # Generate observational data
        obs_data_x = [ToyGraph.X(np.random.normal(0, 1), noise_stdev=1) for x in range(0,1000)]
        obs_data_z = [ToyGraph.Z(x, noise_stdev=1) for x in obs_data_x]
        obs_data_y = [ToyGraph.Y(z, noise_stdev=1) for z in obs_data_z]
        # Add to dataframe
        self.observational_samples['X'] = obs_data_x
        self.observational_samples['Z'] = obs_data_z
        self.observational_samples['Y'] = obs_data_y

    

        

    

