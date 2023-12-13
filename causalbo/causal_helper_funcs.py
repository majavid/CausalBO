from numpy import meshgrid, vstack, float64
from pandas import DataFrame
from scipy.spatial import ConvexHull
import torch

# Create subdictionary containing only keys
def subdict_with_keys(dict: dict, keys: list[str]):
    return {k: dict[k] for k in keys}

# Converts n-dimensional boundaries to hull points for volume of convex hull
def bounds_to_hull_points(domains: dict[list[float]]):
    bounds = domains.values()
    grid = meshgrid(*bounds)
    points = vstack([axis.ravel() for axis in grid]).T
    return points

# = (Vol(C(observational_samples)) / Vol(interventional_domain)) * (n / n_max)
# Discards points with values that may be correct but lie outside of the interventional domain.
def calculate_epsilon(observational_samples: DataFrame, interventional_domain: dict[list[float]], n_max: int):
    n = observational_samples.shape[0]

    # Do not include unobserved confounders in the convex hull.
    df = observational_samples[list(interventional_domain.keys())]

    for k in interventional_domain.keys():
        df = df[ (df[k] >= interventional_domain[k][0])
               & (df[k] <= interventional_domain[k][1])]    
    try:
        epsilon =  (ConvexHull(df).volume / 
                    ConvexHull(
                    bounds_to_hull_points(interventional_domain))
                    .volume) * (n / n_max)
    except:
        # If all points happen to be discarded due to lying outside interventional domain, we need to sample new points.
        epsilon = 1
    
    return epsilon

# Convert DataFrame to torch compatible tensor.
def df_to_tensor(df: DataFrame):
    return torch.tensor(df.to_numpy().astype(float64))


