from causalbo.do_calculus import E_output_given_do, SCM
from causalbo.causal_helper_funcs import calculate_epsilon
import numpy as np
import torch
from matplotlib import pyplot as plt
from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.cbo_loop import CBOLoop

from causalbo.modules import CausalMean, CausalRBF
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model



tg = ToyGraph()

# obs_data = tg.observational_samples[:1000]

# obj_data_x = torch.linspace(-5,5,1000).view(-1,1)
# obj_data_y = tg.obj_func['X'](obj_data_x)

# tg.graph.fit(obs_data)

# xs = np.linspace(-5,5,1000)
# ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=tg.graph) for x in xs]

# plt.plot(obj_data_x, obj_data_y)
# plt.plot(xs, ys)
# plt.show()

from causalbo.modules import CausalMean, CausalRBF
from causalbo.do_calculus import SCM
from causalbo.causal_helper_funcs import *

from pandas import DataFrame

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
# Ground truth functions
def X(input_tensor, noise_mean=0, noise_stdev=0):
    input_tensor = input_tensor[..., :1]
    noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
    return input_tensor + noise

def Z(input_tensor, noise_mean=0, noise_stdev=0):
    input_tensor = input_tensor[..., :1]
    noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
    return (torch.exp(-input_tensor)) + noise

def Y(input_tensor, noise_mean=0, noise_stdev=0):
    input_tensor = input_tensor[..., :1]
    noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
    return ((torch.cos(input_tensor)) - (torch.exp(-input_tensor / 20))) + noise

# Interventional domain = boundaries for each variable.
interventional_domain = {'X': [-5,5], 'Z': [-5,20]}

# Graph = DAG corresponding to causal connections.
graph = SCM([('X', 'Z'), ('Z', 'Y')])

# Store objective functions in a dict for easy access later.
objective_functions = { 'X': lambda x: Y(Z(X(x))),
                        'Z': lambda z: Y(Z(z)),
                        'Y': lambda y: Y(y)}

# Generate some noisy observational data.
obs_data_x = X(torch.randn(1000, 1).view(-1,1), noise_stdev=1)
obs_data_z = Z(obs_data_x, noise_stdev=1)
obs_data_y = Y(obs_data_z, noise_stdev=1)

# Add observations to DataFrame for easy access later.
observational_samples = DataFrame()
observational_samples['X'] = torch.flatten(obs_data_x).tolist()
observational_samples['Z'] = torch.flatten(obs_data_z).tolist()
observational_samples['Y'] = torch.flatten(obs_data_y).tolist()

# Fit observations to DAG to estimate causal effects.
graph.fit(observational_samples)
# Ignore warnings regarding data scaling for simple example
import warnings
warnings.filterwarnings("ignore")

train_x = torch.rand(5, 1)  # Random initial points
train_y = objective_functions['X'](train_x)

# Initialize the GP model
gp = SingleTaskGP(train_x, train_y,
                  covar_module=CausalRBF(
                      interventional_variable=['X'],
                      causal_model=graph
                  ),
                  mean_module=CausalMean(
                      interventional_variable=['X'],
                      causal_model=graph
                  ))

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Initialize the acquisition function (Expected Improvement)
acq_fn = ExpectedImprovement(gp, best_f=0)

# Bayesian optimization loop
num_iterations = 10
for iteration in range(num_iterations):
    # Optimize the acquisition function
    candidate, _ = optimize_acqf(
        acq_function=acq_fn,
        bounds=torch.tensor(list(subdict_with_keys(interventional_domain, ['X']).values()), dtype=torch.float64).t(),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    # Evaluate the objective function at the new candidate point
    new_x = candidate.detach()
    new_y = objective_functions['X'](new_x)

    # Update the training data
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    # Update the GP model and fit the hyperparameters
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # Update the acquisition function with new observations
    best_f = torch.max(train_y)
    acq_fn = ExpectedImprovement(gp, best_f)

# Retrieve the best-performing point
best_point = train_x[torch.argmin(train_y)]

print("Optimal Solution:", best_point.item())


# (global_optimum, global_optimal_set, gp, D_i, D_o) = CBOLoop(
#         observational_samples=tg.observational_samples,
#         graph=tg.graph,
#         exploration_set=[['X']],
#         num_steps=10,
#         num_initial_obs=400,
#         num_obs_per_step=20,
#         num_max_allowed_obs=1000,
#         interventional_domain=tg.interventional_domain,
#         type_trial='max',
#         objective_function=tg.obj_func,
#         early_stopping_iters=2, verbose=False)


test_X = torch.linspace(*tg.interventional_domain['X'], 200, dtype=torch.float64)
obj_func_x = torch.linspace(*tg.interventional_domain['X'], 1000).view(-1,1)
obj_func_y = tg.obj_func['X'](obj_func_x)

D_o = observational_samples

xs = np.arange(-5, 5, 0.01)
ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=graph) for x in xs]

f, ax = plt.subplots(1, 1, figsize=(6, 4))
with torch.no_grad():
    # compute posterior
    posterior = gp.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()

    ax.plot(obj_func_x, obj_func_y, "k")
    # Plot objective
    #plt.axhline(y=global_optimum, color='r', linestyle='-')
    ax.plot(xs, ys, "r")
    # Plot observational data
    ax.scatter(D_o['X'], D_o['Y'])
    # Plot training points as black stars
    ax.plot(gp.train_inputs[0].numpy(), gp.train_targets.numpy(), "k*")
    # Plot posterior means as blue line
    ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5
    )
ax.legend(["Ground truth", "m(x)", "D_O", "D_I", "GP model", "Confidence"])
ax.set_title('Causal GP.')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()