from do_calculus import SCM, E_output_given_do, V_output_given_do
from gpytorch.means.mean import Mean
from gpytorch.kernels import RBFKernel
import torch
import numpy as np


### MEAN FUNC, COVAR KERNEL ###

#TODO: assert input shape matches num interventional nodes
class CausalMean(Mean):
    def __init__(self, interventional_variable: str, causal_model: SCM):
        super(CausalMean, self).__init__()
        self.mean_function = E_output_given_do
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model

    def forward(self, x: torch.Tensor):
        shape = x.shape
        # Flatten input tensor to list of lists for DoWhy to play nice with.
        x_reshape = torch.reshape(x, (-1, shape[-1]))
        mean_output = torch.tensor([
            self.mean_function(
                interventional_variable=self.interventional_variable,
                interventional_value=v.tolist(),
                causal_model=self.causal_model) 
            for v in x_reshape
        ])
        
        # Match input shape and drop lowest dimension.
        output = torch.reshape(mean_output, shape[:-1])

        return output
    

class CausalRBF(RBFKernel):
    # Inherit from base RBFKernel class, add additional information about interventional variable(s) and SCM
    def __init__(self, interventional_variable, causal_model, ard_num_dims=None, batch_shape=None, active_dims=None, lengthscale_prior=None, lengthscale_constraint=None, eps=1e-06, **kwargs):
        super(CausalRBF, self).__init__(ard_num_dims, batch_shape, active_dims, lengthscale_prior, lengthscale_constraint, eps, **kwargs)
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model

    def forward(self, x1, x2, diag=False, **params):
        x1_shape = x1.shape
        x2_shape = x2.shape
        
        # Flatten input tensor to list of lists for DoWhy to play nice with.
        x1_reshape = torch.reshape(x1, (-1, x1.shape[-1]))
        x2_reshape = torch.reshape(x2, (-1, x2.shape[-1]))

        # Calculate variance of each point.
        variances_x1 = torch.sqrt(torch.tensor([
            V_output_given_do(
                interventional_variable=self.interventional_variable,
                interventional_value=x.tolist(),
                causal_model=self.causal_model)
            for x in x1_reshape
        ]))

        variances_x2 = torch.sqrt(torch.tensor([
            V_output_given_do(
                interventional_variable=self.interventional_variable,
                interventional_value=x.tolist(),
                causal_model=self.causal_model)
            for x in x2_reshape
        ]))

        # Match input shape and reduce smallest dimension to 1.
        variances_x1_reshape = torch.reshape(variances_x1, x1_shape[:-1] + (1,))
        variances_x2_reshape = torch.reshape(variances_x2, x2_shape[:-1] + (1,))

        # Create variance matrix of same shape as RBF output.
        variances = (variances_x1_reshape * variances_x2_reshape).transpose(-2,-1)

        # = k_RBF(x_s, x'_s) + sigma(x_s) * sigma(x'_s) 
        return super().forward(x1, x2, diag, **params) + variances