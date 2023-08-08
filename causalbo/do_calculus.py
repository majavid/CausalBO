from networkx import DiGraph, draw
from dowhy import gcm
from numpy import mean, var

### SCM AND DO-CALCULUS ###

class SCM():
    # Declare SCM as nx.DiGraph, auto calculates output node
    def __init__(self, graph: DiGraph | list):
        if type(graph) is DiGraph:
            self.graph = graph
        else:
            try:
                self.graph = DiGraph(graph)
            except:
                raise Exception('Graph must be networkx.DiGraph object or networkx.DiGraph formatted list.')
        self.output_node = [n for n, d in self.graph.out_degree() if d == 0][0]
        self.causal_model = gcm.StructuralCausalModel(self.graph)

    # Fits SCM to observational data by estimating causal mechanisms from DiGraph and data
    def fit(self, observational_samples):
        self.observational_samples = observational_samples
        gcm.auto.assign_causal_mechanisms(self.causal_model, observational_samples)
        gcm.fit(self.causal_model, observational_samples)

    # Perform intervention on node(s) by setting them to value(s)
    def intervene(self, interventional_variable: list[str], interventional_value: list[float]):
        #print(f"{interventional_variable}: {interventional_value} - tensor {input_tensor}")
        intervention_dict = {key: (lambda v: lambda x: v)(value) 
                                   for key, value in zip(interventional_variable, interventional_value)}
        
        samples = gcm.interventional_samples(self.causal_model,
                                             intervention_dict,
                                             observed_data=self.observational_samples)
                                             #num_samples_to_draw=100)
        return samples
    
    # draw graph
    def draw(self):
        draw(self.graph, with_labels=True)

# Expectation given do is average of samples
def E_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM):
    samples = causal_model.intervene(interventional_variable, interventional_value)
    return mean(samples[f'{causal_model.output_node}'])

# Variance given do is variance of samples
def V_output_given_do(interventional_variable: list[str], interventional_value: list[float], causal_model: SCM):
    samples = causal_model.intervene(interventional_variable, interventional_value)
    return var(samples[f'{causal_model.output_node}'])