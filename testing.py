import causal_mean_kernel as cmk
from do_calculus import SCM
from gpytorch.kernels import RBFKernel
import numpy as np
import math
import pandas as pd

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

def test():
    x_bounds = (-5, 5)
    z_bounds = (-5, 20)

    def X(input, noise_mean=0, noise_stdev=0):
        return input + np.random.normal(noise_mean, noise_stdev)

    def Z(input, noise_mean=0, noise_stdev=0):
        return (math.e ** -input) + np.random.normal(noise_mean, noise_stdev)

    def Y(input, noise_mean=0, noise_stdev=0):  
        return ((math.cos(input)) - (math.e ** (-input / 20))) + np.random.normal(noise_mean, noise_stdev)

    obj_func_x = np.linspace(*x_bounds, 1000)
    obj_func_y = [Y(Z(X(x))) for x in obj_func_x]

    obs_data_x = [X(np.random.normal(0, 1), noise_stdev=1) for x in obj_func_x]
    obs_data_z = [Z(x, noise_stdev=1) for x in obs_data_x]
    obs_data_y = [Y(z, noise_stdev=1) for z in obs_data_z]

    observational_samples = pd.DataFrame()
    observational_samples['X'] = obs_data_x
    observational_samples['Z'] = obs_data_z
    observational_samples['Y'] = obs_data_y

    toy_graph = SCM([('X', 'Z'), ('Z', 'Y')])
    toy_graph.fit(observational_samples)

    s = ['X']
    #test_X = torch.stack((torch.linspace(*x_bounds, 200, dtype=torch.float64), torch.linspace(*z_bounds, 200, dtype=torch.float64)), dim=1)
    test_X = torch.linspace(*x_bounds, 200, dtype=torch.float64)
    #initial_intervention_points = [[E_output_given_do(interventional_variable=s, interventional_value=[x] * len(s), causal_model=toy_graph)] for x in [2.0, 0.0, -4.0]]
    initial_intervention_points = [[Y(Z(X(x)))] for x in [2.0, 0.0, -4.0, 4.0]]

    train_X=torch.tensor([[2.0], [0.0], [-4.0], [4.0]], dtype=torch.float64)
    train_Y=torch.tensor(initial_intervention_points, dtype=torch.float64)

    #train_X = torch.stack((torch.tensor(observational_samples['X']), torch.tensor(observational_samples['Z'])), dim=1)
    #train_Y = torch.tensor(observational_samples['Y']).unsqueeze(1)

    model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                                covar_module=cmk.CausalRBF(
                                    interventional_variable=s,
                                    causal_model=toy_graph),
                                mean_module=cmk.CausalMean(
                                    interventional_variable=s,
                                    causal_model=toy_graph))

    # model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
    #                                 covar_module=RBFKernel(),
    #                                 mean_module=ZeroMean())

    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    acqf = ExpectedImprovement(model, best_f=0)
    fit_gpytorch_model(mll)

    candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[-5.0], [5.0]]),
                q=1,
                num_restarts=10,
                raw_samples=50,
            )

test()

# t1 = torch.tensor([[[-1.2131,1]],

#         [[ 1.3041,1]],

#         [[ 3.5492,1]],

#         [[-3.9331,1]],

#         [[-3.7193,1]]], dtype=torch.float64)
# t2 = torch.tensor([[[ 2.0000,1],
#          [ 0.0000,1],
#          [-4.0000,1],
#          [ 4.0000,1],
#          [-1.2131,1]],

#         [[ 2.0000,1],
#          [ 0.0000,1],
#          [-4.0000,1],
#          [ 4.0000,1],
#          [ 1.3041,1]],

#         [[ 2.0000,1],
#          [ 0.0000,1],
#          [-4.0000,1],
#          [ 4.0000,1],
#          [ 3.5492,1]],

#         [[ 2.0000,1],
#          [ 0.0000,1],
#          [-4.0000,1],
#          [ 4.0000,1],
#          [-3.9331,1]],

#         [[ 2.0000,1],
#          [ 0.0000,1],
#          [-4.0000,1],
#          [ 4.0000,1],
#          [-3.7193,1]]], dtype=torch.float64)

# def fake_func(array):
#     return [sum(array)]

# rbf = RBFKernel()
# t1r = torch.reshape(t1, (-1, t1.shape[-1]))
# t2r = torch.reshape(t2, (-1, t2.shape[-1]))

# print(t1r)
# print(t2r)

# t1n = torch.tensor([fake_func(a) for a in t1r])
# t2n = torch.tensor([fake_func(a) for a in t2r])

# print(t1n)
# print(t2n)

# t1nr = torch.reshape(t1n, t1.shape[:-1] + (1,))
# t2nr = torch.reshape(t2n, t2.shape[:-1] + (1,))

# print(t1nr)
# print(t2nr)

# t3 = (t1nr + t2nr).transpose(-2,-1)
# print(t3)

# print(t3.shape)
# print(rbf.forward(t1,t2).shape)

# print

