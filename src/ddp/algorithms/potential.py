import numpy as np

def potential(theta):
    """Simple potential vector: p(theta_i) = theta_i / 2."""
    theta = np.asarray(theta, dtype=float)
    return theta / 2.0
