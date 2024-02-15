import random
import numpy as np

def normal_dist(mu, sigma):
    return lambda: abs(random.normalvariate(mu, sigma))

def uniform_dist(a, b):
    return lambda: random.uniform(a, b)

def gamma_dist(shape, scale):
    return lambda: np.random.gamma(shape, scale)

def half_levy_dist(scale):
    return lambda: abs(np.random.standard_t(2) * scale)

def log_normal_dist(mean, sigma):
    return lambda: np.random.lognormal(mean, sigma)

def t_dist(degrees_of_freedom):
    return lambda: abs(np.random.standard_t(degrees_of_freedom))
