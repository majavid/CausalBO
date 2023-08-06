from causal_mean_kernel import CausalMean, CausalRBF
from do_calculus import SCM
from causal_helper_funcs import calculate_epsilon, df_to_tensor, bounds_to_hull_points, subdict_with_keys
from gpytorch.kernels import RBFKernel
import numpy as np
import math
from pandas import DataFrame
from typing import Literal

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from matplotlib import pyplot as plt

from toy_graph import ToyGraph

tg = ToyGraph()
obj_func_x = np.linspace(*tg.interventional_domain['X'], 1000)
obj_func_y = [tg.obj_func(x) for x in obj_func_x]
plt.scatter(tg.observational_samples['X'], tg.observational_samples['Y'], alpha=0.2)
plt.plot(obj_func_x, obj_func_y, 'k')

def CBOLoop(observational_samples: DataFrame, graph: SCM, exploration_set: list[list[str]], 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, num_max_allowed_obs: int,
            interventional_domain: dict[list[float]], type_trial: Literal['min', 'max'], objective_function):
    
    num_total_obs: int = num_initial_obs
    D_o: DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[DataFrame] = {}
    GPs: dict[SingleTaskGP] = {}
    global_optimum: float

    if type_trial == 'min':
        global_optimum = min(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = max(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return
    
    # Initialize: Set D_i_0 = D_i and D_o_0 = D_o
    for s in exploration_set:
        set_identifier = ''.join(s)
        input_dim = len(s)
        GPs[set_identifier] = SingleTaskGP(train_X=torch.empty(0, input_dim, dtype=torch.float64), train_Y=torch.empty(0, 1, dtype=torch.float64),
                                covar_module=CausalRBF(
                                    output_variable=graph.output_node,
                                    interventional_variable=s,
                                    causal_model=graph),
                                mean_module=CausalMean(
                                    interventional_variable=s,
                                    causal_model=graph))
        D_i[set_identifier]= DataFrame(columns=s + [graph.output_node])
    
    for t in range(num_steps):
        uniform = np.random.uniform(0., 1.)
        if t == 0:
            epsilon = 1
        elif t == 1:
            epsilon = 0
        else:
            epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)

        # Observe
        print(f'Epsilon: {epsilon} - Uniform: {uniform}')
        if(epsilon > uniform):
            print(f'Observing {num_obs_per_step} new observations.')
            num_total_obs += num_obs_per_step
            D_o = observational_samples[:num_total_obs]
            graph.fit(D_o)
        
        else:
            print('Intervening...')
            solutions = {}
            for s in exploration_set:
                set_identifier = ''.join(s)
                gp: SingleTaskGP = GPs[set_identifier]
                interventional_data: DataFrame = D_i[set_identifier]
                if not interventional_data.empty:
                    gp.set_train_data(inputs=df_to_tensor(interventional_data.loc[:, interventional_data.columns != graph.output_node]),
                                      targets=df_to_tensor(interventional_data[graph.output_node]))
                
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    
                    fit_gpytorch_model(mll)
                
                # TODO: Figure out how to run acqf with no training data, or figure out starting interventional data meaning

                acqf = ExpectedImprovement(gp, best_f=global_optimum)
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor([[-5], [5]]), #torch.tensor(list(subdict_with_keys(interventional_domain, s).values())).t(),
                    q=1,
                    num_restarts=10,
                    raw_samples=100
                )
            exit(0)



CBOLoop(observational_samples=tg.observational_samples,
        graph=tg.graph,
        exploration_set=[['X']],
        num_steps=40,
        num_initial_obs=100,
        num_obs_per_step=20,
        num_max_allowed_obs=1000,
        interventional_domain=tg.interventional_domain,
        type_trial='min',
        objective_function=tg.obj_func)

