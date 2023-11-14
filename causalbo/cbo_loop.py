from causalbo.modules import CausalMean, CausalRBF
from causalbo.do_calculus import SCM, E_output_given_do
from causalbo.causal_helper_funcs import calculate_epsilon, df_to_tensor, subdict_with_keys
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
            interventional_domain: dict[list[float]], type_trial: Literal['min', 'max'], objective_function: SCM,
            early_stopping_iters: int = 0, verbose: bool = False):
    
    num_total_obs: int = num_initial_obs
    D_o: DataFrame = observational_samples[:num_initial_obs]
    D_i: dict[DataFrame] = {}
    GPs: dict[SingleTaskGP] = {}
    global_optimum: float = -20
    global_optimal_set: list[str] = ['None']
    global_optimal_value: float = None
    num_iters_without_improvement: int = 0
    total_cost: float = 0
    optimum_over_time: list = []
    cost_over_time: list = []

    # Check if proper type_trial is declared, set current optimum to optimum among observational samples.
    if type_trial == 'min':
        global_optimum = max(D_o[graph.output_node])
    elif type_trial == 'max':
        global_optimum = min(D_o[graph.output_node])
    else:
        print('Invalid type_trial, use either "min" or "max"')
        return
    
    # Initialize: Set D_i_0 = D_i and D_o_0 = D_o
    for s in exploration_set:
        # Create string set identifier for referencing specific subsets of exploration set.
        set_identifier = ''.join(s)
        input_dim = len(s)
        # For each S in ES place GP prior on f(S) = E[Y | do(S = x)]
        GPs[set_identifier] = SingleTaskGP(train_X=torch.empty(0, input_dim, dtype=torch.float64), train_Y=torch.empty(0, 1, dtype=torch.float64),
                                covar_module=CausalRBF(
                                    interventional_variable=s,
                                    causal_model=graph),
                                mean_module=CausalMean(
                                    interventional_variable=s,
                                    causal_model=graph))
        D_i[set_identifier] = DataFrame(columns=s + [graph.output_node])
    
    # Main loop, iterate until num_steps reached or early stopping initiates.
    for t in range(num_steps):
        # Track optimum value found over time for analysis purposes.
        optimum_over_time.append(global_optimum)
        # Early stopping. Will break loop if global optimum does not improve after a specified number of steps.
        if(early_stopping_iters != 0 and num_iters_without_improvement > early_stopping_iters):
            print("Early stopping reached max num of iters without improvment.")
            cost_over_time.append(total_cost)
            break

        print(f"Iteration {t}")
        print(f"Current global optimal set-value-result = {global_optimal_set}: {global_optimal_value} -> {global_optimum}")

        # Decide whether to observe or intervene based on epsilon value - see causal_helper_funcs.calculate_epsilon for more info
        uniform = np.random.uniform(0., 1.)
        if t == 0:
            epsilon = 1
        elif t == 1:
            epsilon = 0
        else:
            epsilon = calculate_epsilon(observational_samples=D_o, interventional_domain=interventional_domain, n_max=num_max_allowed_obs)

        if verbose:
            print(f'Epsilon: {epsilon} - Uniform: {uniform}')

        # Observe
        if(epsilon > uniform):
            print(f'Observing {num_obs_per_step} new data points.')
            num_total_obs += num_obs_per_step
            D_o = observational_samples[:num_total_obs]
            graph.fit(D_o)

            # Fixed cost observation, TODO: Add variable cost
            total_cost += 1
            cost_over_time.append(total_cost)

        # Intervene
        else:
            # Fixed cost intervention, TODO: Add variable cost
            
            
            print('Intervening...')

            solutions = {}

            # Compute EI(S) / Co(S) for each set S in ES
            # TODO: Co(S) means nothing without variable cost intervention. Beyond scope for now?
            for s in exploration_set:
                total_cost += 10 * len(s)
                set_identifier = ''.join(s)
                gp: SingleTaskGP = GPs[set_identifier]
                interventional_data: DataFrame = D_i[set_identifier]
                
                # Build acquisition function depending on max or min result desired.
                if type_trial == 'max':
                    acqf = ExpectedImprovement(gp, best_f=global_optimum)
                else:
                    acqf = ExpectedImprovement(gp, best_f=global_optimum, maximize=False)

                # Generate best predicted candidate sampling point using acqf + mean
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.tensor(list(subdict_with_keys(interventional_domain, s).values()), dtype=torch.float64).t(),
                    q=1,
                    num_restarts=10,
                    raw_samples=100
                )

                # Save best point for set S
                new_x = candidates.detach()
                improvement = acqf.forward(candidates).item()
                solutions[improvement] = (set_identifier, new_x, s)

            # Determine best point for all sets in ES
            best_point = solutions[max(solutions.keys())]
            x_values = torch.flatten(best_point[1]).tolist()
            
            # Perform intervention. Generate ground truth result for sampling at best point.
            # NOTE: "Ground truth" here relies on causal estimation, so may not be 100% accurate. Depends on DoWhy's accuracy. "Ground truth" generated from non-noisy SEM + DoWhy causal model.
            new_y = torch.tensor([ E_output_given_do(interventional_variable=best_point[2], interventional_value=np.array(torch.flatten(best_point[1])), causal_model=objective_function) ])
            if verbose:
                print(f'Optimal set-value pair: {best_point[0]} - {x_values}')

            # This is horrendous, fix later
            if verbose:
                print(f'Updating D_i for {best_point[0]}...')

            columns = [*best_point[2], graph.output_node]
            interventional_data = D_i[best_point[0]]
            interventional_data = concat(
                                [interventional_data, 
                                 DataFrame([dict(zip(columns, x_values + torch.flatten(new_y).tolist()))])
                            ]) 

            if verbose:
                print(f'Updating GP posterior for {best_point[0]}...')

            # Update and retrain GP for chosen best set.
            gp = GPs[best_point[0]]
            gp.set_train_data(inputs=df_to_tensor(interventional_data.loc[:, interventional_data.columns != graph.output_node]),
                              targets=df_to_tensor(interventional_data[graph.output_node]),
                              strict=False)
                      
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            GPs[best_point[0]] = gp
            D_i[best_point[0]] = interventional_data

            cost_over_time.append(total_cost)

            if verbose:
                print('Updating global optimum...')

            # Update global data tracking.
            if (global_optimum == None or 
               (type_trial == 'max' and torch.flatten(new_y)[0] > global_optimum) or 
               (type_trial == 'min' and torch.flatten(new_y)[0] < global_optimum)):
                global_optimum = torch.flatten(new_y)[0]
                global_optimal_set = best_point[2]
                global_optimal_value = torch.flatten(best_point[1]).tolist()
            else:
                num_iters_without_improvement += 1

    return (global_optimum, global_optimal_set, GPs[''.join(global_optimal_set)], D_i[''.join(global_optimal_set)], D_o, cost_over_time, optimum_over_time)



