from causalbo.do_calculus import E_output_given_do, SCM
from causalbo.causal_helper_funcs import calculate_epsilon
import numpy as np
import torch
from matplotlib import pyplot as plt
from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.sample_data.psa_graph import PSAGraph
from causalbo.cbo_loop import CBOLoop

from causalbo.modules import CausalMean, CausalRBF
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

from ananke.graphs import ADMG
from ananke.estimation import CausalEffect
from ananke.estimation import AutomatedIF
from ananke.datasets import load_conditionally_ignorable_data

import matplotlib.pyplot as plt

p = PSAGraph()
exploration_set = [['ASPIRIN', 'STATIN']]
CBOLoop(p.observational_samples,
        p.graph,
        exploration_set,
        40, 400, 40, 1000,
        p.interventional_domain, "min", p.true_graph, early_stopping_iters=10, verbose=True)

# t = ToyGraph()
# exploration_set = [['X', 'Z']]
# CBOLoop(t.observational_samples, t.graph, exploration_set, 40, 100, 40, 1000, t.interventional_domain, "min", t.true_graph, early_stopping_iters=10, verbose=True)


# p.graph.fit(p.observational_samples)
# c = CausalMean(interventional_variable=['ASPIRIN'], causal_model=p.true_graph)
# obj_data_x = torch.linspace(0,1,1000).view(-1,1)
# obj_data_y = c.forward(obj_data_x)

# plt.plot(obj_data_x, obj_data_y)

# plt.show()

# t = ToyGraph()
# exploration_set = [['Z']]

# G = ADMG(['X', 'Z', 'Y'], [('X', 'Z'), ('Z', 'Y')])
# ace_obj = CausalEffect(graph = G, treatment='X', outcome='Y')
# ace = ace_obj.compute_effect(t.objective_samples, "eff-aipw")



#data = load_conditionally_ignorable_data()

# print(t.objective_samples.to_markdown())
# # print(data[data['CD4'] == 0])
# #print(ace)


# import matplotlib.pyplot as plt
# t.graph.fit(t.observational_samples)
# c = CausalMean(interventional_variable=['Z'], causal_model=t.graph)
# obj_data_x = torch.linspace(-5, 20, 2000).view(-1,1)
# obj_data_z = ToyGraph.Y(obj_data_x)
# plt.plot(obj_data_x, obj_data_z)
# plt.plot(obj_data_x, c.forward(obj_data_x))
# plt.show()


#CBOLoop(t.observational_samples, t.graph, exploration_set, 40, 100, 40, 1000, t.interventional_domain, "min", t.true_graph, early_stopping_iters=10, verbose=True)
