from tracemalloc import Statistic
import matplotlib.pyplot as plt
import math
import random
from collections import Counter
from typing import Tuple,List,Callable,TypeVar,Iterator
import random

from numpy import subtract,add
from scipy import datasets

"""the gradient gives the input direction in which the function most quickly increases"""
Vector=List[float]
def difference_quotient(f:Callable[[float],float],x:float,h:float)->float:
    return(f(x+h)-f(x))/h

def square(x:float)->float:
    return x*x

def derivative(x:float)->float:
    return 2*x

xs=range(-10,10)
actuals=[derivative(x) for x in xs]
estimates=[difference_quotient(square,x,h=0.001)for x in xs]

plt.title("Actual Deivatives vs Estimates")
plt.plot(xs,actuals,'rx',label='Actual')
plt.plot(xs,estimates,'b+',label='estimates')
plt.legend(loc=9)
plt.show()


def magnitude(v: Vector) -> float:
    return math.sqrt(sum(v_i ** 2 for v_i in v))
def distance(v:Vector,w:Vector)->float:
    return magnitude(subtract(v,w))

def scalar_multiply(c:float,v:Vector)->Vector:
    return[c*v_i for v_i in v]

def partial_derivative_quotient(f:Callable[[Vector],float],
                                v:Vector,i:int,h:float)->float:
    w=[v_j+(h if j==i else 0) for j,v_j in enumerate(v)]

    #print(f"v: {v}, w: {w}, f(w): {f(w)}, f(v): {f(v)}")

    return (f(w)-f(v))/h

def estimate_gradient(f:Callable[[Vector],float],v:Vector,h:float=0.0001)->Vector:
    gradients=[partial_derivative_quotient(f,v,i,h) for i in range (len(v))]

    #print(f"gradient at {v}:{gradients}")

    return gradients

if __name__=="__main__":
    def example_function(v:Vector)->float:
        return v[0]**2+v[1]**3 #x^2+y^3;at [1.0,2.0]
    v=[1.0,2.0]
     #index of the variable to which the partial derivative is computed, 0 for x and 1 for y
    h=0.001

    #pdq=partial_derivative_quotient(example_function,v,i,h)
    gradient=estimate_gradient(example_function,v,h)
    
    #print(f"Estimated gradient:{gradient}")

    
"""def estimate_gradient(f:Callable[[Vector],float],v:Vector,h:float=0.0001):
    return [partial_derivative_quotient(f,v,i,h) for i in range(len(v))],does the samme work as the above 
    function"""

"""Gradients to find the minimum among all three dimensional vectors"""
def gradient_step(v:Vector,gradient:Vector,step_size:float)->Vector: #step_size is a scalar
    assert len(v)==len(gradient)
    step=scalar_multiply(step_size,gradient)
    return add(v,step)

def sum_of_squares_gradient(v:Vector)->Vector:
    return[2*v_i for v_i in v] #takes a vector v of x^2+y^2+z^2 and returns the gradient

v=[random.uniform(-10,10) for i in range(3)]#Initialize a vector v with three random floats in(-10,10)
for epoch in range(1000): #runs a thousand iterations 
    grad=sum_of_squares_gradient(v) #calculates the gradient
    v=gradient_step(v,grad,-0.01) #updates the position of v by taking a step in the direction opposite to that of the gradient of size 0.01
   # print(epoch,v)

dis=distance(v,[0,0,0]) #distance between final position and the origin  
#print(f"Distance is:{dis}")

    
"""We'll use the gradient descent to find the slope and intercept that minimize the average squared error"""
def linear_gradient(x:float,y:float,theta:Vector)->Vector:
    slope,intercept=theta
    predicted=slope*x+intercept
    error=(predicted-y)
    squared_error=error**2
    grad=[2*error*x,2*error] #compute the gradients with respect to slope and intercept 
    return grad

def vector_mean(xs:Vector)->Vector:
    n=len(xs)
    sum_xs=[sum(x) for x in zip(*xs)]
    return[sx/n for sx in sum_xs]

inputs=[(x,20*x+5)for x in range(-50,50)]

theta=[random.uniform(-1,1),random.uniform(-1,1)] #random values for slope,intercept
learning_rate=0.001
for epoch in range(500):
    grad=vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
    theta=gradient_step(theta,grad,-learning_rate) #updates the position of theta by taking a step in -0.001 direction 
    #print(epoch,theta)
    """the gradient_step functions does (theta-learning_rate*grad)"""
"""MINIBATCH AND STOCHASTIC GRADIENT DESCENT"""

#MINIBATCH Gradient descent **(important)
T=TypeVar('T')

def minibatches(dataset:List[T],batch_size:int,shuffle:bool=True)->Iterator[List[T]]:
    batch_starts=[start for start in range(0,len(dataset),batch_size)]
    if shuffle: random.shuffle(batch_starts)
    for start in batch_starts:
       end=start+batch_size
       yield dataset[start:end]

theta=[random.uniform(-1,1),random.uniform(-1,1)]
for epoch in range(1000):
    for batch in minibatches(inputs,batch_size=20):
     grad=vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
     theta=gradient_step(theta,grad,-learning_rate) #updates the position of theta by taking a step in -0.001 direction 
     #print(epoch,theta)
        

    