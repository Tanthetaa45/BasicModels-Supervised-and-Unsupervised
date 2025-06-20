import random
from typing import List, Callable, TypeVar, Tuple
from Stats import standard_deviation
import datetime
from MLR import sqerror_gradient, vector_mean, gradient_step
import tqdm
from functools import partial

# Type Definitions
Vector = List[float]
X = TypeVar('X')
Stat = TypeVar('Stat')

# Least Squares Fit Function
def least_squares_fit(xs: List[Vector], ys: List[float], learning_rate: float = 0.001,
                      num_steps: int = 1000, batch_size: int = 1) -> Vector:
    """Finds the beta that minimizes the sum of squared errors assuming the model y = dot(x, beta)"""
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least_squares_fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

# Median Function
def median(data: List[float]) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    else:
        return sorted_data[mid]

# Bootstrap Sampling and Statistic Functions
def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X], stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

# Function to Estimate Beta Using Least Squares Fit
def estimate_sample_beta(pairs: List[Tuple[Vector, float]], learning_rate: float) -> Vector:
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    return beta

# Seed for Reproducibility
random.seed(0)

# Sample Input Data
inputs = [
    [1, 2, 3],  # Example vector 1
    [4, 5, 6],  # Example vector 2
    [7, 8, 9],  # Example vector 3
]

# Normalize Daily Minutes Good
daily_minutes_good = [50, 60, 70, 80, 90]
daily_minutes_good = [(x - min(daily_minutes_good)) / (max(daily_minutes_good) - min(daily_minutes_good)) for x in daily_minutes_good]

# Set the Learning Rate
learning_rate = 0.01

# Bootstrap Betas Using the Specified Learning Rate
bootstrap_betas = bootstrap_statistic(
    list(zip(inputs, daily_minutes_good)),
    lambda pairs: estimate_sample_beta(pairs, learning_rate),
    100
)

# Calculate Bootstrap Standard Errors
bootstrap_standard_errors = [standard_deviation([beta[i] for beta in bootstrap_betas])
                             for i in range(len(inputs[0]))]  # Adjusted to match the input vector size

# Print the Results
print(bootstrap_standard_errors)
