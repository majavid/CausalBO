from typing import Literal

from botorch.models import SingleTaskGP
from pandas import DataFrame
from do_calculus import SCM
import causal_helper_funcs
import causal_mean_kernel
import torch

from causal_mean_kernel import CausalMean, CausalRBF

def CBOLoop(observational_samples: DataFrame, graph: SCM, exploration_set: list[list[str]], 
            num_steps: int, num_initial_obs: int, num_obs_per_step: int, num_max_allowed_obs: int,
            interventional_domain: list[list[float]], type_trial: Literal['min', 'max'], objective_function: function):
    
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
        GPs[set_identifier] = SingleTaskGP(train_X=torch.empty(0, input_dim), train_Y=torch.empty(0, 1),
                                covar_module=CausalRBF(
                                    output_variable=graph.output_node,
                                    interventional_variable=s,
                                    causal_model=graph),
                                mean_module=CausalMean(
                                    interventional_variable=s,
                                    causal_model=graph))
        D_i[set_identifier]= DataFrame(columns=s + graph.output_node)



