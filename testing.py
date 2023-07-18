import networkx as nx
from dowhy import gcm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from random import random
from time import time

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel
from gpytorch.means.mean import Mean
from gpytorch.means import ZeroMean
import torch
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch.optim import SGD

### SCM AND DO-CALCULUS ###

class SCM():
    # Declare SCM as nx.DiGraph, auto calculates output node
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.output_node = [n for n, d in self.graph.out_degree() if d == 0][0]
        self.causal_model = gcm.StructuralCausalModel(self.graph)

    # Fits SCM to observational data by estimating causal mechanisms from DiGraph and data
    def fit(self, observational_samples):
        self.observational_samples = observational_samples
        gcm.auto.assign_causal_mechanisms(self.causal_model, observational_samples)
        gcm.fit(self.causal_model, observational_samples)

    # Perform intervention on node(s) by setting them to value(s)
    def intervene(self, interventional_variable: list[str], interventional_value: list[float]):
        intervention_dict = {key: (lambda v: lambda x: v)(value) 
                                   for key, value in zip(interventional_variable, interventional_value)}
        samples = gcm.interventional_samples(self.causal_model,
                                             intervention_dict,
                                             observed_data=self.observational_samples)
                                             # num_samples_to_draw=1000)
        return samples

# Expectation given do is average of samples
def E_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM):
    samples = causal_model.intervene(interventional_variable, interventional_value)
    return np.mean(samples[f'{causal_model.output_node}'])

# Variance given do is variance of samples
def V_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM):
    samples = causal_model.intervene(interventional_variable, interventional_value)
    return np.var(samples[f'{causal_model.output_node}'])

### MEAN FUNC, COVAR KERNEL ###
    
class CausalMean(Mean):
    # Set mean function, interventional variable(s), and SCM
    def __init__(self, interventional_variable: str, causal_model: SCM):
        super(CausalMean, self).__init__()
        self.mean_function = E_output_given_do
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model

    # Output smoothed results of mean function.
    def forward(self, x: torch.Tensor):
        return torch.tensor(savgol_filter(
                                [self.mean_function(
                                interventional_variable=self.interventional_variable,
                                causal_model=self.causal_model,
                                interventional_value=v.tolist()) for v in x],
                            window_length=2, polyorder=1, mode='interp'),
                        dtype=torch.float64)

# As CausalMean but without smoothing    
class CausalMeanUnsmoothed(Mean):
    def __init__(self, interventional_variable: str, causal_model: SCM):
        super(CausalMeanUnsmoothed, self).__init__()
        self.mean_function = E_output_given_do
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model

    def forward(self, x: torch.Tensor):
        return torch.tensor([self.mean_function(
                                interventional_variable=self.interventional_variable,
                                causal_model=self.causal_model,
                                interventional_value=v.tolist()) for v in x])
    
class CausalRBF(RBFKernel):
    # Inherit from base RBFKernel class, add additional information about interventional variable(s) and SCM
    def __init__(self, interventional_variable, causal_model, ard_num_dims=None, batch_shape=None, active_dims=None, lengthscale_prior=None, lengthscale_constraint=None, eps=1e-06, **kwargs):
        super(CausalRBF, self).__init__(ard_num_dims, batch_shape, active_dims, lengthscale_prior, lengthscale_constraint, eps, **kwargs)
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model

    # = k_RBF(x_s, x'_s) + sigma(x_s) * sigma(x'_s)
    def forward(self, x1, x2, diag=False, **params):
        variances_x1 = np.sqrt(np.array([
            V_output_given_do(interventional_variable=self.interventional_variable,
                              interventional_value=x1_i.tolist(),
                              causal_model=self.causal_model)
            for x1_i in x1
        ]))

        variances_x2 = np.sqrt(np.array([
            V_output_given_do(interventional_variable=self.interventional_variable,
                              interventional_value=x2_j.tolist(),
                              causal_model=self.causal_model)
            for x2_j in x2
        ]))

        outer_product = np.outer(variances_x1, variances_x2)

        return super().forward(x1, x2, diag, **params) + torch.tensor(outer_product, dtype=torch.float64)
    
                        

### IMPLEMENTATION ###
# Converts 2D x/y boundaries to hull points for volume of convex hull
def bounds_to_hull_points(x_bounds: tuple, y_bounds: tuple):
    return [[x_bounds[0], y_bounds[0]], [x_bounds[0], y_bounds[1]], [x_bounds[1], y_bounds[0]], [x_bounds[1], y_bounds[1]]]

# = (Vol(C(observational_samples)) / Vol(interventional_domain)) * (n / n_max)
def calculate_epsilon(observational_samples: pd.DataFrame, interventional_domain, n, n_max):
    epsilon = (ConvexHull(observational_samples).volume / 
               ConvexHull(
                    bounds_to_hull_points(interventional_domain[0], interventional_domain[1]))
                .volume) * (n / n_max)
    return epsilon

# Mostly pseudocode and structuring example IGNORE
def CBOLoop(observational_samples: pd.DataFrame, graph: SCM, exploration_set: list[list[str]], 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, interventional_domain,
            num_allowed_obs, type_trial: str, objective_function):
    
    num_total_obs: int = num_initial_obs
    D_o: pd.DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[pd.DataFrame] = {}
    GPs: dict[SingleTaskGP] = {}
    global_optimum: float = 0

    if type_trial == 'min':
        global_optimum = min(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = max(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return
    
    for s in exploration_set:
        set_identifier = ''.join(s)
        input_dim = len(s)
        #inital_intervention_point = E_output_given_do(interventional_variable=s, interventional_value=[0.0] * len(s), causal_model=graph)
        GPs[set_identifier] = SingleTaskGP(train_X=torch.empty(0, input_dim), train_Y=torch.empty(0, 1),
                                covar_module=CausalRBF(
                                    output_variable=graph.output_node,
                                    interventional_variable=s,
                                    causal_model=graph),
                                mean_module=CausalMean(
                                    interventional_variable=s,
                                    causal_model=graph))
        D_i[set_identifier]= pd.DataFrame()

    for t in range(num_steps):
        uniform = np.random.uniform(0., 1.)
        if t == 0:
            epsilon = 1
        elif t == 1:
            epsilon = 0
        else:
            epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain)

        if(epsilon > uniform):
            num_total_obs += num_obs_per_step
            D_o = observational_samples[:num_total_obs]
            graph.fit(D_o)
        else:
            improvements = {}
            for s in exploration_set:
                set_identifier = ''.join(s)
                gp: SingleTaskGP = GPs[set_identifier]
                gp.set_train_data(torch.tensor(inputs = D_i[set_identifier][s], targets = D_i[set_identifier][graph.output_node]))
                acqf = ExpectedImprovement(gp)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor(bounds_to_hull_points(*interventional_domain)),
                    q=1,
                    num_restarts=10,
                    raw_samples=100
                )

                new_x = candidates.detach()[0].item()
                improvement = acqf.forward(candidates).item()
                improvements[set_identifier] = {'improvement': improvement, 'new_x': new_x, 'set_identifier': set_identifier}
            
            best_entry = max(improvements.values(), key=lambda x: x['improvement'])

            D_i[best_entry[set_identifier]] = D_i[best_entry[set_identifier]].append({'Z': new_x, 'Y': objective_function(new_x)})
            #GPs[best_entry[set_identifier]].set_train_data()

    return GPs, D_i

# CBOLoop stripped down for debugging purposes
def timing_graph(observational_samples: pd.DataFrame, graph: SCM, exploration_set: list[list[str]], 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, interventional_domain,
            num_allowed_obs, type_trial: str, objective_function):
    
    num_total_obs: int = num_initial_obs
    D_o: pd.DataFrame = observational_samples[:num_initial_obs]
    D_i: pd.DataFrame
    Gp: SingleTaskGP
    global_optimum: float = 0
    global_cost: float = 0
    global_optimum_over_time = []

    if type_trial == 'min':
        global_optimum = min(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = max(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return
    
    # Initialize GP for interventional variable Z, SCM X->Z->Y
    Gp = SingleTaskGP(train_X=torch.empty(0, 1), train_Y=torch.empty(0, 1),
                                covar_module=CausalRBF(
                                    output_variable=graph.output_node,
                                    interventional_variable=['Z'],
                                    causal_model=graph),
                                mean_module=CausalMeanUnsmoothed(
                                    interventional_variable=['Z'],
                                    causal_model=graph))
    # Empty data frame for interventions
    D_i = pd.DataFrame()

    for t in range(num_steps):
        # Always observe once, then intervene once, then calc epsilon
        uniform = np.random.uniform(0., 1.)
        if t == 0:
            epsilon = 1
        elif t == 1:
            epsilon = 0
        else:
            epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain)

        # Observe
        if(epsilon > uniform):
            # This code runs fine. Observe, then fit graph.
            num_total_obs += num_obs_per_step
            D_o = observational_samples[:num_total_obs]
            graph.fit(D_o)
        # Intervene
        else:
            global_cost += 1
            if global_cost != 1:
                # Set training data of GP to D_I
                Gp.set_train_data(torch.tensor(inputs = D_i['Z'], targets = D_i['Y']))
            # Standard EI acquisition function
            acqf = ExpectedImprovement(Gp, torch.tensor(global_optimum))

            # Error shows up here
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[-5.0], [20.0]]),
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            # Evaluate the objective function at the new candidate points
            new_x = candidates.detach()
            new_y = objective_function(new_x)

            # Update the training data
            D_i.append({'Z': new_x, 'Y': new_y})

            # Update the GP model and fit the hyperparameters
            Gp.set_train_data(torch.tensor(inputs = D_i['Z'], targets = D_i['Y']))
            mll = ExactMarginalLogLikelihood(Gp.likelihood, Gp)
            fit_gpytorch_model(mll)

            # Update the acquisition function with new observations. Not necessary since other things are happening between optim runs?
            acqf.update(Gp)

        # Update global optimum
        if type_trial == 'min':
            try:
                global_optimum = min(D_i[graph.output_node])
            except:
                global_optimum = global_optimum
        elif type_trial == 'max':
            try:
                global_optimum = max(D_i[graph.output_node])
            except:
                global_optimum = global_optimum
        global_optimum_over_time.append([global_optimum, global_cost])

    return global_optimum_over_time


# Execution portion of code
x_bounds = (-5, 5)
z_bounds = (-5, 20)

# Objective funcs
def X(input, noise_mean=0, noise_stdev=0):
    return input + np.random.normal(noise_mean, noise_stdev)

def Z(input, noise_mean=0, noise_stdev=0):
    return (math.e ** -input) + np.random.normal(noise_mean, noise_stdev)

def Y(input, noise_mean=0, noise_stdev=0):  
    return ((math.cos(input)) - (math.e ** (-input / 20))) + np.random.normal(noise_mean, noise_stdev)

obj_func_x = np.linspace(*x_bounds, 1000)
obj_func_y = [Y(Z(X(x))) for x in obj_func_x]

# Generate observational data
obs_data_x = [X(np.random.normal(0, 1), noise_stdev=1) for x in obj_func_x]
obs_data_z = [Z(x, noise_stdev=1) for x in obs_data_x]
obs_data_y = [Y(z, noise_stdev=1) for z in obs_data_z]

observational_samples = pd.DataFrame()
observational_samples['X'] = obs_data_x
observational_samples['Z'] = obs_data_z
observational_samples['Y'] = obs_data_y

# Declare graph
toy_graph = SCM(nx.DiGraph([('X', 'Z'), ('Z', 'Y')]))

timing_graph(observational_samples=observational_samples,
        graph=toy_graph,
        exploration_set=[['Z']],
        num_steps=40,
        num_initial_obs=100,
        num_obs_per_step=20,
        interventional_domain=[x_bounds, z_bounds],
        num_allowed_obs=1000,
        type_trial='min',
        objective_function=lambda x: Y(Z(x)))

## OLD TESTING CODE KEPT AROUND TO POTENTIALLY REFERENCE LATER? QUICKER THAN STACKOVERFLOW ##

# def test():
#     x_bounds = (-5, 5)
#     z_bounds = (-5, 20)

#     def X(input, noise_mean=0, noise_stdev=0):
#         return input + np.random.normal(noise_mean, noise_stdev)

#     def Z(input, noise_mean=0, noise_stdev=0):
#         return (math.e ** -input) + np.random.normal(noise_mean, noise_stdev)

#     def Y(input, noise_mean=0, noise_stdev=0):  
#         return ((math.cos(input)) - (math.e ** (-input / 20))) + np.random.normal(noise_mean, noise_stdev)

#     obj_func_x = np.linspace(*x_bounds, 1000)
#     obj_func_y = [Y(Z(X(x))) for x in obj_func_x]


#     obs_data_x = [X(np.random.normal(0, 1), noise_stdev=1) for x in obj_func_x]
#     obs_data_z = [Z(x, noise_stdev=1) for x in obs_data_x]
#     obs_data_y = [Y(z, noise_stdev=1) for z in obs_data_z]

#     observational_samples = pd.DataFrame()
#     observational_samples['X'] = obs_data_x
#     observational_samples['Z'] = obs_data_z
#     observational_samples['Y'] = obs_data_y

#     toy_graph = SCM(nx.DiGraph([('X', 'Z'), ('Z', 'Y')]))
#     # observational_samples = pd.DataFrame(pd.read_pickle('./Data/ToyGraph/observations.pkl'))


#     # plt.scatter(observational_samples['X'], observational_samples['Y'])
#     # plt.show()
#     toy_graph.fit(observational_samples)

#     intervention_cost = 1

#     s = ['X']
#     #test_X = torch.stack((torch.linspace(*x_bounds, 200, dtype=torch.float64), torch.linspace(*z_bounds, 200, dtype=torch.float64)), dim=1)
#     test_X = torch.linspace(*x_bounds, 200, dtype=torch.float64)
#     #initial_intervention_points = [[E_output_given_do(interventional_variable=s, interventional_value=[x] * len(s), causal_model=toy_graph)] for x in [2.0, 0.0, -4.0]]
#     initial_intervention_points = [[Y(Z(X(x)))] for x in [2.0, 0.0, -4.0, 4.0]]

#     train_X=torch.tensor([[2.0], [0.0], [-4.0], [4.0]], dtype=torch.float64)
#     train_Y=torch.tensor(initial_intervention_points, dtype=torch.float64)

#     #train_X = torch.stack((torch.tensor(observational_samples['X']), torch.tensor(observational_samples['Z'])), dim=1)
#     #train_Y = torch.tensor(observational_samples['Y']).unsqueeze(1)

#     model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
#                                 covar_module=CausalRBF(
#                                     output_variable=toy_graph.output_node,
#                                     interventional_variable=s,
#                                     causal_model=toy_graph),
#                                 mean_module=CausalMean(
#                                     interventional_variable=s,
#                                     causal_model=toy_graph))

#     # model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
#     #                                 covar_module=RBFKernel(),
#     #                                 mean_module=ZeroMean())

#     mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
#     fit_gpytorch_model(mll)
#     # mll = mll.to(train_X)

#     # optimizer = SGD([{"params": model.parameters()}], lr=0.1)

#     # NUM_EPOCHS = 150

#     # model.train()
#     # mll.train()

#     # for epoch in range(NUM_EPOCHS):
#     #     # clear gradients
#     #     optimizer.zero_grad()
#     #     # forward pass through the model to obtain the output MultivariateNormal
#     #     output = model(train_X)
#     #     # Compute negative marginal log likelihood
#     #     loss = -mll(output, model.train_targets)
#     #     # back prop gradients
#     #     loss.backward()
#     #     # print every 10 iterations
#     #     optimizer.step()

#     # model.eval()
#     # mll.eval()

#     xs = np.arange(-5, 5, 0.01)
#     ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=toy_graph) for x in xs]


#     f, ax = plt.subplots(1, 1, figsize=(6, 4))
#     with torch.no_grad():
#         # compute posterior
#         posterior = model.posterior(test_X)
#         # Get upper and lower confidence bounds (2 standard deviations from the mean)
#         lower, upper = posterior.mvn.confidence_region()

#         ax.plot(obj_func_x, obj_func_y, "k")
#         # Plot objective
#         #plt.axhline(y=0, color='r', linestyle='-')
#         ax.plot(xs, ys, "r")
#         # Plot observational data
#         ax.scatter(observational_samples['X'], observational_samples['Y'])
#         # Plot training points as black stars
#         ax.plot(train_X.cpu().numpy(), train_Y.cpu().numpy(), "k*")
#         # Plot posterior means as blue line
#         ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), "b")
#         # Shade between the lower and upper confidence bounds
#         ax.fill_between(
#             test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5
#         )
#     ax.legend(["Ground truth", "m(x)", "D_O", "D_I", "GP model", "Confidence"])
#     ax.set_title('Causal GP.')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.tight_layout()
#     plt.show()