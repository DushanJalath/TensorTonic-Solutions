import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.array(points)
    assignments = np.array(assignments)

    return [points[assignments==i].mean(axis=0).tolist() for i in range(k)]