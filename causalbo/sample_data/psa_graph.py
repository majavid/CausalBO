from pandas import DataFrame
from causalbo.do_calculus import SCM
import torch

# Sample DAG and SCM using medical dataset provided by V. Aglietti et al.
# CausalBO does not require data to be organized in this fashion, but it does help to keep it organized in a similar manner.
class PSAGraph(object):
    # U(55, 75)
    def age(num_data_points):
        return (55 - 75) * torch.rand(num_data_points, 1) + 75
    
    # N(27.0 − 0.01 × age, 0.7)
    def bmi(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return torch.normal((torch.full_like(input_tensor, 27.) - 0.01 * input_tensor), 0.7) + noise

    # σ(−8.0 + 0.10 × age + 0.03 × bmi)
    def aspirin(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[-8 + 0.1 * i[0] + 0.03 * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return torch.nn.Sigmoid()(new_tensor) + noise

    # σ(−13.0 + 0.10 × age + 0.20 × bmi)
    def statin(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :2]
        new_tensor = torch.tensor([[-13 + 0.1 * i[0] + 0.2 * i[1]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return torch.nn.Sigmoid()(new_tensor) + noise
    
    # σ(2.2 − 0.05 × age + 0.01 × bmi − 0.04 × statin + 0.02 × aspirin)
    def cancer(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :4]
        new_tensor = torch.tensor([[2.2 - 0.05 * i[0] + 0.01 * i[1] - 0.04 * i[2] + 0.02 * i[3]] for i in input_tensor])
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return torch.nn.Sigmoid()(new_tensor) + noise
    
    #N (6.8 + 0.04 × age − 0.15 × bmi − 0.60 × statin + 0.55 × aspirin + 1.00 × cancer, 0.4)
    def psa(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :5]
        new_tensor = torch.normal(torch.tensor([[6.8 + 0.04 * i[0] - 0.15 * i[1] - 0.60 * i[2] + 0.55 * i[3] + 1.00 * i[4]] for i in input_tensor]), 0.4)
        noise = torch.normal(noise_mean, noise_stdev, new_tensor.shape)
        return new_tensor + noise
    
    def __init__(self, num_observations = 1000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        # Interventional domain
        self.interventional_domain = {'ASPIRIN': [0,1], 'STATIN': [0,1]}
        # Graph structure
        self.graph = SCM([('AGE', 'BMI'),
                          ('AGE', 'ASPIRIN'),
                          ('AGE', 'PSA'),
                          ('AGE', 'CANCER'),
                          ('AGE', 'STATIN'),

                          ('BMI', 'ASPIRIN'),
                          ('BMI', 'CANCER'),
                          ('BMI', 'STATIN'),
                          ('BMI', 'PSA'),

                          ('ASPIRIN', 'CANCER'),
                          ('ASPIRIN', 'PSA'),

                          ('STATIN', 'CANCER'),
                          ('STATIN', 'PSA'),

                          ('CANCER', 'PSA')
                          ])
        
        # Same structure, deep copy
        self.true_graph = SCM([('AGE', 'BMI'),
                          ('AGE', 'ASPIRIN'),
                          ('AGE', 'PSA'),
                          ('AGE', 'CANCER'),
                          ('AGE', 'STATIN'),

                          ('BMI', 'ASPIRIN'),
                          ('BMI', 'CANCER'),
                          ('BMI', 'STATIN'),
                          ('BMI', 'PSA'),

                          ('ASPIRIN', 'CANCER'),
                          ('ASPIRIN', 'PSA'),

                          ('STATIN', 'CANCER'),
                          ('STATIN', 'PSA'),

                          ('CANCER', 'PSA')
                          ])

        # Generate observational data
        obs_data_age = PSAGraph.age(num_observations)
        obs_data_bmi = PSAGraph.bmi(obs_data_age, noise_mean=0, noise_stdev=0.2)
        obs_data_aspirin = PSAGraph.aspirin(torch.cat([obs_data_age, obs_data_bmi], dim=1), noise_mean=0, noise_stdev=0.2)
        obs_data_statin = PSAGraph.statin(torch.cat([obs_data_age, obs_data_bmi], dim=1), noise_mean=0, noise_stdev=0.2)
        obs_data_cancer = PSAGraph.cancer(torch.cat([obs_data_age, obs_data_bmi, obs_data_statin, obs_data_aspirin], dim=1), noise_mean=0, noise_stdev=0.2)
        obs_data_psa = PSAGraph.psa(torch.cat([obs_data_age, obs_data_bmi, obs_data_statin, obs_data_aspirin, obs_data_cancer], dim=1), noise_mean=0, noise_stdev=0.2)

        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['AGE'] = torch.flatten(obs_data_age).tolist()
        self.observational_samples['BMI'] = torch.flatten(obs_data_bmi).tolist()
        self.observational_samples['ASPIRIN'] = torch.flatten(obs_data_aspirin).tolist()
        self.observational_samples['STATIN'] = torch.flatten(obs_data_statin).tolist()
        self.observational_samples['CANCER'] = torch.flatten(obs_data_cancer).tolist()
        self.observational_samples['PSA'] = torch.flatten(obs_data_psa).tolist()

        # Remove invalid samples caused by randomness
        self.observational_samples = self.observational_samples.drop(
                                        self.observational_samples[self.observational_samples['PSA'] <= 0].index)
        
        # Fit graph to observational data.
        self.graph.fit(self.observational_samples)

        # Generate objective data
        obs_data_age = PSAGraph.age(num_objective_points)
        obs_data_bmi = PSAGraph.bmi(obs_data_age)
        obs_data_aspirin = PSAGraph.aspirin(torch.cat([obs_data_age, obs_data_bmi], dim=1))
        obs_data_statin = PSAGraph.statin(torch.cat([obs_data_age, obs_data_bmi], dim=1))
        obs_data_cancer = PSAGraph.cancer(torch.cat([obs_data_age, obs_data_bmi, obs_data_statin, obs_data_aspirin], dim=1))
        obs_data_psa = PSAGraph.psa(torch.cat([obs_data_age, obs_data_bmi, obs_data_statin, obs_data_aspirin, obs_data_cancer], dim=1))

        # # Add to dataframe
        self.objective_samples = DataFrame()
        self.objective_samples['AGE'] = torch.flatten(obs_data_age).tolist()
        self.objective_samples['BMI'] = torch.flatten(obs_data_bmi).tolist()
        self.objective_samples['ASPIRIN'] = torch.flatten(obs_data_aspirin).tolist()
        self.objective_samples['STATIN'] = torch.flatten(obs_data_statin).tolist()
        self.objective_samples['CANCER'] = torch.flatten(obs_data_cancer).tolist()
        self.objective_samples['PSA'] = torch.flatten(obs_data_psa).tolist()

        # Remove invalid samples caused by randomness
        self.objective_samples = self.objective_samples.drop(
                                        self.objective_samples[self.objective_samples['PSA'] <= 0].index)

        # Fit graph to objective data.
        self.true_graph.fit(self.objective_samples)        

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

