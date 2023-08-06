from numpy import meshgrid, vstack
from pandas import DataFrame
from scipy.spatial import ConvexHull


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
def calculate_epsilon(observational_samples: DataFrame, interventional_domain: dict[list[float]], n_max: int):
    n = observational_samples.shape[0]
    epsilon = (ConvexHull(observational_samples).volume / 
               ConvexHull(
                    bounds_to_hull_points(interventional_domain))
                .volume) * (n / n_max)
    return epsilon

# Convert DataFrame to torch compatible tensor.
def df_to_tensor(df: DataFrame):
    return tensor(df.to_numpy())


