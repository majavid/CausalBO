from causal_mean_kernel import CausalMean, CausalRBF
from do_calculus import SCM
from causal_helper_funcs import calculate_epsilon, df_to_tensor, subdict_with_keys
import numpy as np
from pandas import DataFrame, concat
from typing import Literal

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

def CBOLoop(observational_samples: DataFrame, graph: SCM, exploration_set: list[list[str]], 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, num_max_allowed_obs: int,
            interventional_domain: dict[list[float]], type_trial: Literal['min', 'max'], objective_function: dict):
    
    num_total_obs: int = num_initial_obs
    D_o: DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[DataFrame] = {}
    GPs: dict[SingleTaskGP] = {}
    global_optimum: float = -20
    global_optimal_set: str = 'None'

    # if type_trial == 'min':
    #     global_optimum = min(D_o[graph.output_node])
    # elif type_trial == 'max':
    #     global_optimum = max(D_o[graph.output_node])
    # else:
    #     print('Invalid type_trial, use either "min" or "max"')
    #     return
    
    # Initialize: Set D_i_0 = D_i and D_o_0 = D_o
    for s in exploration_set:
        set_identifier = ''.join(s)
        input_dim = len(s)
        GPs[set_identifier] = SingleTaskGP(train_X=torch.empty(0, input_dim, dtype=torch.float64), train_Y=torch.empty(0, 1, dtype=torch.float64),
                                #covar_module=TestKernel(),
                                covar_module=CausalRBF(
                                    interventional_variable=s,
                                    causal_model=graph),
                                mean_module=CausalMean(
                                    interventional_variable=s,
                                    causal_model=graph))
        D_i[set_identifier]= DataFrame(columns=s + [graph.output_node])
    
    for t in range(num_steps):
        print(f"Iteration {t}")
        print(f"Current global optimum: {global_optimum}")
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
            print(f'Observing {num_obs_per_step} new data points.')
            num_total_obs += num_obs_per_step
            D_o = observational_samples[:num_total_obs]
            graph.fit(D_o)
        # Intervene
        else:
            print('Intervening...')
            solutions = {}
            for s in exploration_set:
                set_identifier = ''.join(s)
                gp: SingleTaskGP = GPs[set_identifier]
                interventional_data: DataFrame = D_i[set_identifier]
                
                acqf = ExpectedImprovement(gp, best_f=global_optimum)
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor(list(subdict_with_keys(interventional_domain, s).values()), dtype=torch.float64).t(),
                    q=1,
                    num_restarts=10,
                    raw_samples=100
                )

                new_x = candidates.detach()
                improvement = acqf.forward(candidates).item()
                solutions[improvement] = (set_identifier, new_x)

            best_point = solutions[max(solutions.keys())]

            #TODO: Figure out objective functions for evey possible set in ES
            x_values = torch.flatten(best_point[1]).tolist()
            
            new_y = objective_function[best_point[0]](best_point[1])

            print(f'Optimal set-value pair: {best_point[0]} - {x_values}')

            columns = [*best_point[0], graph.output_node]

            # This is horrendous, fix later
            print(f'Updating D_i for {best_point[0]}...')
            interventional_data = concat(
                                [interventional_data, 
                                 DataFrame([dict(zip(columns, x_values + torch.flatten(new_y).tolist()))])
                            ]) 

            print(f'Updating GP posterior for {best_point[0]}...')
            gp.set_train_data(inputs=df_to_tensor(interventional_data.loc[:, interventional_data.columns != graph.output_node]),
                              targets=df_to_tensor(interventional_data[graph.output_node]),
                              strict=False)
            
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            GPs[best_point[0]] = gp
            D_i[best_point[0]] = interventional_data


            print('Updating global optimum...')
            if(global_optimum == None or torch.flatten(new_y)[0] > global_optimum):
                global_optimum = torch.flatten(new_y)[0]
                global_optimal_set = best_point[0]

    return (global_optimum, global_optimal_set, GPs[global_optimal_set], D_i[global_optimal_set], D_o)



