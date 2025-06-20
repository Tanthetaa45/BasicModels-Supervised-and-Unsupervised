from typing import List
import math
import numpy as np



Vector=List[float]
height_weight_age=[70,170,40]
grades=[95,80,75,62]

from typing import List

# Define Vector as a type alias for a list of floats (or ints if preferred)
Vector = List[float]

def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v, w)]

# Test the function
result = subtract([5, 7, 9], [4, 5, 6])
#print(result)  # This will print: [1, 2, 3]
#suports vector addition,subtraction,multiplication by a scalar,dot product..

def dot(x:Vector,y:Vector)->float:
    assert len(x)==len(y)
    return sum(x_i*y_i for x_i,y_i in zip(x,y))
result1=dot([1,2,3],[4,5,6])
#print(result1) #gives the dot product, and for magnitude ,dot the vector with itself
#magnitude
def magnitude(v:Vector)->float:
    return math.sqrt(sum(v_i**2 for v_i in v))
vector=[3,4]
result=magnitude(vector)
#print(result)



# Define Vector as a type alias for a list of floats (or ints if preferred)
Vector = List[float]

def magnitude(v: Vector) -> float:
    return math.sqrt(sum(v_i ** 2 for v_i in v))

def magnitudes(vectors: List[Vector]) -> List[float]:
    return [magnitude(v) for v in vectors]

# Test the function
vectors = [
    [3, 4],
    [1, 2, 2],
    [5, 12]
]

results = magnitudes(vectors)
#print(results) # for multiple vectors

def distance(v:Vector,w:Vector)->float:
    return magnitude(subtract(v,w))
vector1=([3,5])
vector2=([4,7])

result=distance(vector1,vector2)
#print(result)