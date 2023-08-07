from do_calculus import E_output_given_do
from causal_helper_funcs import calculate_epsilon
import numpy as np
import torch
from matplotlib import pyplot as plt
from toy_graph import ToyGraph
from cbo_loop import CBOLoop

tg = ToyGraph()
# tg.graph.fit(tg.observational_samples)

# xs = np.arange(-5, 5, 0.01)
# ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=tg.graph) for x in xs]


# plt.plot(xs, ys)
# plt.show()
# obs_data = tg.observational_samples[:100]
# epsilon = calculate_epsilon(obs_data, tg.interventional_domain, 1000)
# print(epsilon)


(global_optimum, global_optimal_set, gp, D_i, D_o) = CBOLoop(
        observational_samples=tg.observational_samples,
        graph=tg.graph,
        exploration_set=[['X']],
        num_steps=10,
        num_initial_obs=100,
        num_obs_per_step=20,
        num_max_allowed_obs=1000,
        interventional_domain=tg.interventional_domain,
        type_trial='max',
        objective_function=tg.obj_func)

test_X = torch.linspace(*tg.interventional_domain['X'], 200, dtype=torch.float64)
obj_func_x = torch.linspace(*tg.interventional_domain['X'], 1000).view(-1,1)
obj_func_y = tg.obj_func['X'](obj_func_x)

xs = np.arange(-5, 5, 0.01)
ys = [E_output_given_do(interventional_variable=['X'], interventional_value=[x], causal_model=tg.graph) for x in xs]

f, ax = plt.subplots(1, 1, figsize=(6, 4))
with torch.no_grad():
    # compute posterior
    posterior = gp.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()

    ax.plot(obj_func_x, obj_func_y, "k")
    # Plot objective
    plt.axhline(y=global_optimum, color='r', linestyle='-')
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
ax.legend(["Ground truth", "global_optimum", "m(x)", "D_O", "D_I", "GP model", "Confidence"])
ax.set_title('Causal GP.')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()