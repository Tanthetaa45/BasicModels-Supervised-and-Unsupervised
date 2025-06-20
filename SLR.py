from typing import List, Tuple
import math, random
import tqdm
import numpy as np
from Stats import correlation, standard_deviation, mean
from GradientDescent import gradient_step

# Set the seed for reproducibility
random.seed(0)
np.random.seed(0)

# Number of data points
num_data_points = 100

# Generate random data for num_friends_good and daily_minutes_good
num_friends_good = [random.randint(1, 100) for _ in range(num_data_points)]
daily_minutes_good = [random.randint(30, 300) for _ in range(num_data_points)]

# Normalize the data
num_friends_good = [(x - min(num_friends_good)) / (max(num_friends_good) - min(num_friends_good)) for x in num_friends_good]
daily_minutes_good = [(x - min(daily_minutes_good)) / (max(daily_minutes_good) - min(daily_minutes_good)) for x in daily_minutes_good]

Vector = List[float]

def vector_mean(xs: List[Vector]) -> Vector:
    """Compute the mean of each dimension across all vectors."""
    num_vectors = len(xs)
    num_dimensions = len(xs[0])
    return [sum(vector[i] for vector in xs) / num_vectors for i in range(num_dimensions)]

def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def de_mean(data: List[Vector]) -> List[Vector]:
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def total_sum_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))

# Initialize parameters
num_epochs = 10000
learning_rate = 0.000001  # Reduced learning rate

# Initialize guess
guess = [random.random(), random.random()]

# Perform gradient descent
with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        grad_a = sum(2 * error(alpha, beta, x_i, y_i) for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i for x_i, y_i in zip(num_friends_good, daily_minutes_good))

        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.6f}")

        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

alpha, beta = guess

print(f'Final alpha: {alpha}')
print(f'Final beta: {beta}')
