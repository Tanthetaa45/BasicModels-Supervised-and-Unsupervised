"""Perceptrons are the simplest neural network,it computes a weighted sum of its inputs and 'fires' if that sum is 0 or greater"""
import numpy as np
import pandas as pd
from typing import List
import math
Vector=List[float]


def dot(x:Vector,y:Vector)->float:
    assert len(x)==len(y)
    return sum(x_i*y_i for x_i,y_i in zip(x,y))

def step_function(x:float)->float:
    return 1.0 if x>=0 else 0.0

def perceptron_output(weights:Vector,bias:float,x:Vector)->float:
    calculation=dot(weights,x)+bias
    return step_function(calculation)
and_weights=[2,2]
and_bias=-3
print(f"Perceptron acts as an AND gate:{perceptron_output(and_weights,and_bias,[1,1])}")
#In the above statement we make the perceptron act as an AND gate 
or_weights=[2,2]
or_bias=-1
print(f"Perceptron acts as an OR gate:{perceptron_output(or_weights,or_bias,[1,1])}")
#In the above statement the perceptron acts as an OR gate 
not_weights=[-2.]
not_bias=1.
print(f"The perceptron acts as a NOT gate:{perceptron_output(not_weights,not_bias,[1])}")
"""Feed-forward neural network consists of discrete layers of neurons,each connected to the next"""

def sigmoid(t:float)->float:
    return 1/(1+math.exp(-t))

"""We use the sigmoid function instead of the step_function because the step_function is not continuous whereas the sigmoid function
is continuous and a good approximiation of step_function"""
def neuron_output(weights:Vector,inputs:Vector)->float:
    return sigmoid(dot(weights,inputs))
def feed_forward(neural_network:List[List[Vector]],
                 input_vector:Vector)->List[Vector]:
    outputs:List[Vector]=[]

    for layer in neural_network:
        input_with_bias=input_vector+[1]
        output=[neuron_output(neuron,input_with_bias)
                for neuron in layer]
        outputs.append(output)

        input_vector=output

    return outputs
xor_network=[[[20.,20,-30],[20.,20,-10]],[[-60.,60,-30]]]

print(f"The Feed_forward output is:{feed_forward(xor_network,[1,0])[-1][0]}")
#XOR gates give 1 when either of the inputs are one
def sqerror_gradients(network:List[List[Vector]],input_vector:Vector,target_vector:Vector)->List[List[Vector]]:
    hidden_outputs,outputs=feed_forward(network,input_vector)
    output_deltas=[output*(1-output)*(output-target) for output ,target in zip(outputs,target_vector)]
    output_grads=[[output_deltas[i]*hidden_output for hidden_output in hidden_outputs +[1]]
                  for i,output_neuron in enumerate(network[-1])]
    hidden_deltas=[hidden_output*(1-hidden_output)*dot(output_deltas,
                   [n[i]for n in network[-1]]) for i , hidden_output in 
                   enumerate(hidden_outputs)]
    hidden_grads=[[hidden_deltas[i]*input for input in input_vector+[1]]
                  for i,hidden_neuron in enumerate(network[0])]
    return [hidden_grads,output_grads]