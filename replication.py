from causalbo.do_calculus import E_output_given_do, SCM
from causalbo.causal_helper_funcs import calculate_epsilon, subdict_with_keys, df_to_tensor
import numpy as np
import torch
from matplotlib import pyplot as plt
from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.sample_data.psa_graph import PSAGraph
from causalbo.cbo_loop import CBOLoop

from causalbo.modules import CausalMean, CausalRBF
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

NUM_INITIAL_OBSERVATIONS = 400 
INTERVENTION_COST = 10
OBSERVATION_COST = 1

toy_graph = PSAGraph()

num_initial_interventions = 1
num_iterations = 10

train_x_standard = df_to_tensor(toy_graph.observational_samples.loc[:NUM_INITIAL_OBSERVATIONS][['ASPIRIN', 'STATIN']])
train_y_standard = df_to_tensor(toy_graph.observational_samples.loc[:NUM_INITIAL_OBSERVATIONS,toy_graph.observational_samples.columns == toy_graph.graph.output_node])

# Store total cost
total_cost_standard = NUM_INITIAL_OBSERVATIONS
# Store optimal value
global_optimum_standard = max(toy_graph.observational_samples[toy_graph.graph.output_node])
# Store changes in cost over time
cost_over_time_standard = []
# Store optimum over time
global_optimum_over_time_standard = []
# Standard intervention set is all non-output nodes
intervention_set_standard = ['ASPIRIN', 'STATIN']#toy_graph.observational_samples.loc[:,toy_graph.observational_samples.columns != toy_graph.graph.output_node].columns.tolist()


# Initialize GP

gp_standard = SingleTaskGP(train_X=train_x_standard, train_Y=train_y_standard)
mll = ExactMarginalLogLikelihood(gp_standard.likelihood, gp_standard)
acqf = ExpectedImprovement(model=gp_standard, best_f=global_optimum_standard, maximize=False)
fit_gpytorch_model(mll)

global_optimum_over_time_standard.append(global_optimum_standard)
cost_over_time_standard.append(total_cost_standard)

true_ys = torch.tensor([[]]) #torch.empty_like(train_y_standard)
#print(true_ys)

# Optimization loop
for i in range(num_iterations):
    print(f'Standard GP, iteration {i}')
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=torch.tensor(list(subdict_with_keys(toy_graph.interventional_domain, intervention_set_standard).values()), dtype=torch.float64).t(),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    

    # Evaluate the objective function at the new candidate point
    new_x = candidate.detach()
    new_y = torch.tensor([[E_output_given_do(interventional_variable=intervention_set_standard, interventional_value=np.array(torch.flatten(new_x)), causal_model=toy_graph.true_graph)]])
    print(new_y)

    # Update the training data
    train_x_standard = torch.cat([train_x_standard, new_x])
    train_y_standard = torch.cat([train_y_standard, new_y])

    true_ys = torch.cat([true_ys, new_y], dim=1)
    #print(true_ys)


    # Update the GP model and fit the hyperparameters
    gp_standard = SingleTaskGP(train_x_standard, train_y_standard)
    mll = ExactMarginalLogLikelihood(gp_standard.likelihood, gp_standard)
    fit_gpytorch_model(mll)

    # Update the acquisition function with new observations
    global_optimum_standard = torch.min(true_ys)
    acqf = ExpectedImprovement(model=gp_standard, best_f=global_optimum_standard, maximize=False)

    # Update cost and optimum
    total_cost_standard += INTERVENTION_COST * len(intervention_set_standard)
    cost_over_time_standard.append(total_cost_standard)
    global_optimum_over_time_standard.append(global_optimum_standard)

print(global_optimum_over_time_standard)


# Causal GP
(global_optimum, global_optimal_set, gp, D_i, D_o, cost_over_time_causal, global_optimum_over_time_causal) = CBOLoop(
        observational_samples=toy_graph.observational_samples,
        graph=toy_graph.graph,
        exploration_set=[['ASPIRIN', 'STATIN']], # We are allowed to examine fewer variables here since we know the POMIS is ['Z'] and the causal GP can take advantage of this, while the standard cannot
        num_steps=10,
        num_initial_obs=NUM_INITIAL_OBSERVATIONS,
        num_obs_per_step=20,
        num_max_allowed_obs=1000,
        interventional_domain=toy_graph.interventional_domain,
        type_trial='min',
        objective_function=toy_graph.true_graph,
        early_stopping_iters=10, verbose=True)


plt.plot([0] + cost_over_time_standard, [global_optimum_over_time_standard[0]] + global_optimum_over_time_standard, "-o")

plt.plot(cost_over_time_causal, global_optimum_over_time_causal, "-o")

plt.plot()

plt.legend(['Standard GP', 'Causal GP'])

plt.xlabel("Total Cost\nObservation costs 1 unit per point, intervention costs 10 units per variable")
plt.ylabel("Global Optimum")
plt.title("Standard GP vs Causal GP: Medical Graph, 10 iterations")
plt.show()

# xs = np.arange(-5, 20, 0.1)
# toy_graph.graph.fit(toy_graph.observational_samples)
# ys = [E_output_given_do(interventional_variable=['Z'], interventional_value=[x], causal_model=toy_graph.graph) for x in xs]

obj_func_x = torch.linspace(*toy_graph.interventional_domain['Z'], 1000).view(-1,1)
obj_func_y = toy_graph.obj_func['X'](obj_func_x)
true_optimum = torch.min(obj_func_y)
plt.axhline(true_optimum, color='r')
#plt.plot(obj_func_x, obj_func_y)
#plt.plot(xs,ys)
#plt.scatter(train_input_standard[:,0], train_y_standard)
plt.plot(cost_over_time_standard, global_optimum_over_time_standard)
# plt.plot(cost_over_time_causal, global_optimum_over_time_causal)
plt.title('Causal GP vs Standard GP performance')
plt.xlabel('Total Cost')
plt.ylabel('Calculated Global Optimum')
plt.legend(['True global optimum', 'Standard GP', 'Causal GP'])
plt.show()