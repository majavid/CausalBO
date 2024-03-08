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

NUM_INITIAL_OBSERVATIONS = 40
INTERVENTION_COST = 10
OBSERVATION_COST = 1

toy_graph = PSAGraph(num_objective_points=5000)

num_initial_interventions = 1
num_iterations = 25

train_x_standard = df_to_tensor(toy_graph.observational_samples.loc[:1][['ASPIRIN', 'STATIN']])
train_y_standard = df_to_tensor(toy_graph.observational_samples.loc[:1,toy_graph.observational_samples.columns == toy_graph.graph.output_node])

# Store total cost
total_cost_standard = 0
# Store optimal value
global_optimum_standard = max(toy_graph.observational_samples[toy_graph.graph.output_node][:NUM_INITIAL_OBSERVATIONS])
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



true_ys = torch.tensor([[global_optimum_standard]]) #torch.empty_like(train_y_standard)
#print(true_ys)

# Optimization loop
for i in range(num_iterations):
    global_optimum_over_time_standard.append(global_optimum_standard)
    cost_over_time_standard.append(total_cost_standard)
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
    print(new_x)
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

global_optimum_over_time_standard.append(global_optimum_standard)
cost_over_time_standard.append(total_cost_standard)
print(global_optimum_over_time_standard)


# xs = np.linspace(-5,5,1000)
# xs = torch.tensor(xs).view(-1,1)
# ys = ToyGraph.Y(ToyGraph.Z(ToyGraph.X(xs)))
true_optimum_standard = global_optimum_standard * 0.9#min(toy_graph.objective_samples['PSA'])

#fig, ax = plt.subplots(1, 1)
# ax[0].plot(xs, ys, )
# ax[0].hlines(global_optimum_standard,-5,5,color='red')
# ax[0].hlines(true_optimum_standard,-5,5,color="orange")
# ax[0].xlabel("X")
# ax[0].ylabel("Y")


idx = global_optimum_over_time_standard.index(global_optimum_standard) + 1

global_optimum_over_time_standard = global_optimum_over_time_standard[:idx]
cost_over_time_standard = cost_over_time_standard[:idx]

plt.plot(cost_over_time_standard, global_optimum_over_time_standard, "-o", color='blue', label='Standard GP')

plt.xlabel("Total Cost")
plt.ylabel("Global Optimum")

#plt.show()


# Causal GP
(global_optimum, global_optimal_set, gp, D_i, D_o, cost_over_time_causal, global_optimum_over_time_causal) = CBOLoop(
        observational_samples=toy_graph.observational_samples,
        graph=toy_graph.graph,
        exploration_set=[['ASPIRIN', 'STATIN']], # We are allowed to examine fewer variables here since we know the POMIS is ['Z'] and the causal GP can take advantage of this, while the standard cannot
        num_steps=num_iterations,
        num_initial_obs=40,#NUM_INITIAL_OBSERVATIONS,
        num_obs_per_step=10,
        num_max_allowed_obs=1000,
        interventional_domain=toy_graph.interventional_domain,
        type_trial='min',
        objective_function=toy_graph.true_graph,
        early_stopping_iters=8, verbose=True)

# zs = np.linspace(-5,20,1000)
# zs = torch.tensor(zs).view(-1,1)
# ys = ToyGraphModified.Y(zs)
# true_optimum_standard = min(ys)
# print(true_optimum_standard)

# cm = CausalMean(interventional_variable=['Z'], causal_model=toy_graph.graph)

# es = cm.forward(zs)

# plt.plot(zs, ys)
# plt.plot(zs, es)
# plt.show()

print(global_optimum)
print(global_optimum_over_time_causal)

idx = global_optimum_over_time_causal.index(global_optimum) + 1
global_optimum_over_time_causal = [global_optimum_over_time_causal[0]] + global_optimum_over_time_causal[:idx]
cost_over_time_causal = [0] + cost_over_time_causal[:idx]

plt.plot(cost_over_time_causal, global_optimum_over_time_causal, "-o", color='orange', label='Causal GP')
#plt.hlines(true_optimum_standard,0,cost_over_time_causal[-1],color="red")
plt.hlines(true_optimum_standard,0,max(cost_over_time_standard[-1], cost_over_time_causal[-1]),color="red", label='True Optimum')

plt.legend()

plt.show()