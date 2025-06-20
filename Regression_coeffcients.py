import numpy as np 
import random,math
import matplotlib.pyplot as plt
from typing import List,Tuple
Vector=List[float]
def error(x: Vector, y: float, beta: Vector) -> float:
    
    predicted = dot(x, beta)
    return y - predicted

def add(v:Vector,w:Vector)->Vector:
    assert len(v)==len(w)
    return [v_i+w_i for v_i,w_i in zip(v,w)]

def normal_cdf(x:float,mu:float=0,sigma:float=1)->float:
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma))/2

def p_value(beta_hat_j:float,sigma_hat_j:float)->float:
    if beta_hat_j>0:
        return 2*(1-normal_cdf(beta_hat_j/sigma_hat_j))
    else:
        return 2*normal_cdf(beta_hat_j/sigma_hat_j)
    

print(f"the p_value is:{p_value(0.923,1.249)}")
"""Regularization is an approach in which we add to the error term a penalty 
that gets larger as beta gets larger"""

"""ridge regression"""
def dot(x:Vector,y:Vector)->float:
    assert len(x)==len(y)
    return sum(x_i*y_i for x_i,y_i in zip(x,y))
def ridge_penalty(beta:Vector,alpha:float)->float:
    return alpha*dot(beta[1:],beta[1:])

def squared_error_ridge(x:Vector,y:float,beta:Vector,alpha:float)->float:
    return error(x,y,beta)**2+ridge_penalty(beta,alpha)

def ridge_penalty_gradient(beta:Vector,alpha:float)->Vector:
    return[0.]+[2*alpha*beta_j for beta_j in beta[1:]]

