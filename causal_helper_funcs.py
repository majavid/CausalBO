from numpy import meshgrid, vstack
from pandas import DataFrame
from scipy.spatial import ConvexHull
from torch import tensor

# Converts n-dimensional boundaries to hull points for volume of convex hull
def bounds_to_hull_points(bounds: list):
    grid = meshgrid(*bounds)
    points = vstack([axis.ravel() for axis in grid]).T
    return points.tolist()

# = (Vol(C(observational_samples)) / Vol(interventional_domain)) * (n / n_max)
def calculate_epsilon(observational_samples: DataFrame, interventional_domain, n, n_max):
    epsilon = (ConvexHull(observational_samples).volume / 
               ConvexHull(
                    bounds_to_hull_points(interventional_domain))
                .volume) * (n / n_max)
    return epsilon

# Convert DataFrame to torch compatible tensor.
def df_to_tensor(df: DataFrame):
    return tensor(df.to_numpy())


