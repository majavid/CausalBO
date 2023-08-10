from causalbo.do_calculus import E_output_given_do, SCM
from causalbo.causal_helper_funcs import calculate_epsilon
import numpy as np
import torch
from matplotlib import pyplot as plt
from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.sample_data.synthetic_graph import SyntheticGraph
from causalbo.cbo_loop import CBOLoop

from causalbo.modules import CausalMean, CausalRBF
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


ages = SyntheticGraph.age(10)
bmi = SyntheticGraph.bmi(ages)
aspirin = SyntheticGraph.aspirin(torch.cat([ages, bmi], dim=1))
print(ages)
print(bmi)
print(aspirin)