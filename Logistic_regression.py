import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from typing import List,Tuple,TypeVar
import random,math
import tqdm
from Rescaling import rescale
from MLR import least_squares_fit,predict
from GradientDescent import gradient_step
Vector=List[float]
"""data = [
    [5, 150000, 1],
    [3, 80000, 0],
    [10, 200000, 1],
    [7, 120000, 0]
    # Add more rows as needed
]"""
X=TypeVar('X')

def split_data(data:List[X],prob:float)->Tuple[List[X],List[X]]:
    data=data[:]
    random.shuffle(data)
    cut=int(len(data)*prob)
    return data[:cut],data[cut:]

data=[n for n in range(1000)]
train,test=split_data(data,0.75)



Y = TypeVar('Y')



xs = [[1.0, float(row), float(row) / 2.0] for row in data]  # Add a bias term, a feature, and another feature
ys = [1 if row % 2 == 0 else 0 for row in data] 
def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])


learning_rate=0.001
rescaled_xs=rescale(xs)
beta=least_squares_fit(rescaled_xs,ys,learning_rate,1000,1)
predictions=[predict(x_i,beta) for x_i in rescaled_xs]

plt.scatter(predictions,ys)
plt.xlabel("predicted")
plt.ylabel("actual")
#plt.show()


#For a logistic function as input gets larger and positive the value gets closer and closer to 1
#As input gets larger and negative ,it gets closer to closer to 0
def logistic(x:float)->float:
    return 1.0/(1+math.exp(-x))
def logistic_prime(x:float)->float:
    y=logistic(x)
    return y*(1-y)
def dot(x:Vector,y:Vector)->float:
    assert len(x)==len(y)
    return sum(x_i*y_i for x_i,y_i in zip(x,y))

"""We want to minimize the negative log likelihood which is equal to maximizing the log 
likelihood """
def _negative_log_likelihood(x:Vector,y:float,beta:Vector)->float:
    """The negative log likelihood for one data point"""
    if y==1:
        return -math.log(logistic(dot(x,beta)))
    else:
        return -math.log(1-logistic(dot(x,beta)))
    
def negative_log_likelihood(xs:List[Vector],ys:List[float],beta:Vector)->float:
    return sum(_negative_log_likelihood(x,y,beta) for x,y in zip(xs,ys))

from typing import List, Tuple

def vector_sum(xs: List[float], ys: List[float]) -> List[float]:
    """
    Computes the element-wise sum of two vectors.
    
    Args:
        xs (List[float]): First vector.
        ys (List[float]): Second vector.
    
    Returns:
        List[float]: Element-wise sum of xs and ys.
    """
    if len(xs) != len(ys):
        raise ValueError("Vectors must have the same length")
    
    return [x + y for x, y in zip(xs, ys)]

"""For calculating the gradient"""

def _negative_log_partial_j(x:Vector,y:float,beta:Vector,j:int)->float:
    return -(y-logistic(dot(x,beta)))*x[j]

def _negative_log_gradient(x:Vector,y:float,beta:Vector)->Vector:
    """The gradient for one data poiont"""
    return [_negative_log_partial_j(x,y,beta,j) for j in range(len(beta))]

def negative_log_gradient(xs: List[Vector], ys: List[float], beta: Vector) -> Vector:
    """Computes the total gradient of the negative log-likelihood."""
    # Initialize a zero vector for the sum of gradients
    total_gradient = [0.0] * len(beta)
    
    # Sum gradients from each data point
    for x, y in zip(xs, ys):
        gradient = _negative_log_gradient(x, y, beta)
        total_gradient = vector_sum(total_gradient, gradient)
    
    return total_gradient

nll_value = negative_log_likelihood(rescaled_xs, ys, beta)
#print(f"The NLL is: {nll_value:.4f}")

"""Applying the model"""

random.seed(0)
x_train,x_test,y_train,y_test=train_test_split(rescaled_xs,ys,0.33)
learning_rate=0.01

beta=[random.random() for _ in range(3)]
with tqdm.trange(5000) as t:
    for epoch in t:
        gradient=negative_log_gradient(x_train,y_train,beta)
        beta=gradient_step(beta,gradient,-learning_rate)
        loss=negative_log_likelihood(x_train,y_train,beta)
        t.set_description(f"loss:{loss:.3f} beta:{beta}")

#print(f"the gradient_step goes:{beta}") #[-0.00922013 0.89725024 1.13507486]

true_positives=true_negatives=false_positives=false_negatives=0

for x_i,y_i in zip(x_test,y_test):
    prediction=logistic(dot(beta,x_i))

    if y_i==1 and prediction>=0.5:#paid and predicted paid
        true_positives+=1
    elif y_i==1:
        false_negatives+=1 #paid but predicted unpaid 
    elif prediction>=0.5:
        false_positives+=1 #unpaid but we predicted paid
    else:
        true_negatives+=1 #unpaid and we predicted unpaid 

precision=true_positives/(true_positives+false_positives) #times we predicted paid
recall=true_positives/(true_positives+false_negatives) #times he actually paid

print(f"The recall is:{recall}")

predictions=[logistic(dot(beta,x_i)) for x_i in x_test]
plt.scatter(predictions,y_test,marker='+',color='g')
plt.xlabel("Predicted probability")
plt.ylabel("actual Outcome")
plt.title("Predicted Logistic Regression vs Actual")
plt.show()

