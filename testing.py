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


t = ToyGraph()
exploration_set = [['X']]

CBOLoop(t.observational_samples, t.graph, exploration_set, 40, 100, 40, 1000, t.interventional_domain, "min", t.obj_func, early_stopping_iters=10, verbose=True)
