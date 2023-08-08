from do_calculus import E_output_given_do
from causal_helper_funcs import calculate_epsilon
import numpy as np
import torch
from matplotlib import pyplot as plt
from toy_graph import ToyGraph
from cbo_loop import CBOLoop

from causal_mean_kernel import CausalMean, CausalRBF
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


(global_optimum, global_optimal_set, gp, D_i, D_o) = CBOLoop(
        observational_samples=tg.observational_samples,
        graph=tg.graph,
        exploration_set=[['X']],
        num_steps=10,
        num_initial_obs=400,
        num_obs_per_step=20,
        num_max_allowed_obs=1000,
        interventional_domain=tg.interventional_domain,
        type_trial='max',
        objective_function=tg.obj_func,
        early_stopping_iters=2)



# train_X=torch.tensor([[2.0], [0.0], [-4.0], [4.0]], dtype=torch.float64)
# train_Y = tg.obj_func['X'](torch.tensor([[2.0], [0.0], [-4.0], [4.0]], dtype=torch.float64))


# gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
#                             covar_module=CausalRBF(
#                                 output_variable=tg.graph.output_node,
#                                 interventional_variable=s,
#                                 causal_model=tg.graph),
#                             mean_module=CausalMean(
#                                 interventional_variable=s,
#                                 causal_model=tg.graph))

#     # model = SingleTaskGP(train_X=train_X, train_Y=train_Y,
#     #                                 covar_module=RBFKernel(),
#     #                                 mean_module=ZeroMean())

# mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
# fit_gpytorch_model(mll)

# test_X = torch.linspace(*tg.interventional_domain['X'], 200, dtype=torch.float64)
# obj_func_x = torch.linspace(*tg.interventional_domain['X'], 1000).view(-1,1)
# obj_func_y = tg.obj_func['X'](obj_func_x)

# # D_o = tg.observational_samples

# xs = np.arange(-5, 5, 0.01)
# ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=tg.graph) for x in xs]

# f, ax = plt.subplots(1, 1, figsize=(6, 4))
# with torch.no_grad():
#     # compute posterior
#     posterior = gp.posterior(test_X)
#     # Get upper and lower confidence bounds (2 standard deviations from the mean)
#     lower, upper = posterior.mvn.confidence_region()

#     ax.plot(obj_func_x, obj_func_y, "k")
#     # Plot objective
#     #plt.axhline(y=global_optimum, color='r', linestyle='-')
#     ax.plot(xs, ys, "r")
#     # Plot observational data
#     ax.scatter(D_o['X'], D_o['Y'])
#     # Plot training points as black stars
#     ax.plot(gp.train_inputs[0].numpy(), gp.train_targets.numpy(), "k*")
#     # Plot posterior means as blue line
#     ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), "b")
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(
#         test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5
#     )
# ax.legend(["Ground truth", "m(x)", "D_O", "D_I", "GP model", "Confidence"])
# ax.set_title('Causal GP.')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.tight_layout()
# plt.show()