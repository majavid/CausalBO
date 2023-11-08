from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.modules import CausalMean
import torch
import pandas as pd
import matplotlib.pyplot as plt

graph = ToyGraph()
observational_samples = pd.DataFrame()

obj_data_x = ToyGraph.X(torch.linspace(-5, 5, 2000).view(-1,1))
#obs_data_z = [ToyGraph.Z(x, noise_stdev=1) for x in obs_data_x]
obj_data_z = ToyGraph.Z(obj_data_x)
#obs_data_y = [ToyGraph.Y(z, noise_stdev=1) for z in obs_data_z]
obj_data_y = ToyGraph.Y(obj_data_z)
        # Add to dataframe
observational_samples['X'] = torch.flatten(obj_data_x).tolist()
observational_samples['Z'] = torch.flatten(obj_data_z).tolist()
observational_samples['Y'] = torch.flatten(obj_data_y).tolist()

graph.graph.fit(observational_samples)

cm = CausalMean(interventional_variable=['X'], causal_model=graph.graph)

output = cm.forward(obj_data_x)
plt.plot(obj_data_x, output)
plt.show()
##TODO : REPLACE OBJECTIVE FUNCTION WITH OBJECTIVE FUNCTION APPROXIMATION LIKE SO USING OBJ DATA FIT TO GRAPH