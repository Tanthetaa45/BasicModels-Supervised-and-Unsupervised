from typing import List, Tuple,TypeVar,Callable
import math, random
import tqdm
import numpy as np
from Stats import correlation, standard_deviation, mean
from GradientDescent import gradient_step
from vector import dot


Vector=List[float]

def vector_mean(xs: List[Vector]) -> Vector:
    """Compute the mean of each dimension across all vectors."""
    num_vectors = len(xs)
    num_dimensions = len(xs[0])
    return [sum(vector[i] for vector in xs) / num_vectors for i in range(num_dimensions)]

def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def de_mean(data: List[Vector]) -> List[Vector]:
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

def total_sum_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))


def predict(x:Vector,beta:Vector)->float:
    """Assumes that the first element of x is 1"""

    return dot(x,beta)

"""beta1 = [0.5, 1.2, -0.9]  # Example beta coefficients
x_i = [1, 2, 3]          # Example input vector (the first element is 1 for the intercept)

# Prediction
prediction = predict(x_i, beta1)
print(f"Prediction: {prediction}")"""


def error(x:Vector,y:float,beta:Vector)-> float:
    return predict(x,beta)-y

def squared_error(x:Vector,y:float,beta:Vector)->float:
    return error(x,y,beta)**2

x=[1,2,3]
y=30
beta=[4,4,4]
#print(f"The error is:{error(x,y,beta)}") #-6
#print(f"The squared errors are:{squared_error(x,y,beta)}")

def sqerror_gradient(x:Vector,y:float,beta:Vector)->Vector:
    err=error(x,y,beta)
    return[2*err*x_i for x_i in x]

#print(f"The gradient of error is:{sqerror_gradient(x,y,beta)}")

def least_squares_fit(xs:List[Vector],ys:List[float],learning_rate:float=0.001,num_steps:int=1000,
                      batch_size:int=1)->Vector:
    """it finds the beta that minimises the sum of squared errors assuming the model y=dot(x,beta)"""
    guess=[random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps,desc="least_squares_fit"):
        for start in range(0,len(xs),batch_size):
            batch_xs=xs[start:start+batch_size]
            batch_ys=ys[start:start+batch_size]

            gradient=vector_mean([sqerror_gradient(x,y,guess)
                                  for x,y in zip(batch_xs,batch_ys)]
            )
            guess=gradient_step(guess,gradient,-learning_rate)

    return guess
daily_minutes_good=[50,60,70,80,90]
num_friends_good=[]
num_friends_good = [(x - min(num_friends_good)) / (max(num_friends_good) - min(num_friends_good)) for x in num_friends_good]
daily_minutes_good = [(x - min(daily_minutes_good)) / (max(daily_minutes_good) - min(daily_minutes_good)) for x in daily_minutes_good]
inputs = [1, 2, 3],    # Example vector 1
[4, 5, 6],    # Example vector 2
[7, 8, 9],    #example Vector 3

random.seed(0)
learning_rate=0.001
beta=least_squares_fit(inputs,daily_minutes_good,learning_rate,5000,25)

#print(f"the value of beta:{beta}")

def multiple_r_squared(xs:List[Vector],ys:Vector,beta:Vector)->float:
    sum_of_squared_errors=sum(error(x,y,beta)**2 for x,y in zip(xs,ys))
    return 1.0-sum_of_squared_errors/total_sum_squares(ys)

#print(f"The coeffcient of determination is:{multiple_r_squared(inputs,daily_minutes_good,beta)}")

import random
from typing import List, Callable, TypeVar

X = TypeVar('X')
Stat = TypeVar('Stat')

def median(data: List[float]) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
    else:
        return sorted_data[mid]

def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X], stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

# Sample data
close_to_100 = [99.5 + random.random() for _ in range(101)]

far_from_100 = ([99.5 + random.random()] + [random.random() for _ in range(50)] +
               [200 + random.random() for _ in range(50)])

# Calculate the bootstrap medians
medians_far = bootstrap_statistic(far_from_100, median, 100)
medians_close=bootstrap_statistic(close_to_100,median,100)

#print(f"goo``:{standard_deviation(medians_far)}")
