"""(height,weight) pairs are treated as points in two dimensions"""
"""when height is measured in inches ,the nearest neigbour of B is A but if the height is 
measured in centimetres the nearest neighbour of B is C for which we use rescaling so that 
each dimension has mean as 0 and standard deviation 1"""
from vector import magnitude,distance,dot,subtract
from typing import Tuple,List
from Stats import standard_deviation
import tqdm
import random
from GradientDescent import gradient_step,scalar_multiply
import matplotlib.pyplot as plt
import numpy as np

Vector = List[float]

def vector_mean(xs: List[Vector]) -> Vector:
    """Compute the mean of each dimension across all vectors."""
    num_vectors = len(xs)
    num_dimensions = len(xs[0])
    return [sum(vector[i] for vector in xs) / num_vectors for i in range(num_dimensions)]

a_to_b = distance([63, 150], [67, 160])
a_to_c = distance([63, 150], [70, 171])
b_to_c = distance([67, 160], [70, 171])
#print(f"DISTANCE IS: {a_to_b}")

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(dim)]
    return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]] #(-3-1+1)/3
means, stdevs = scale(vectors)

#print(f"The means are: {means}")
#print(f"The standard deviations are: {stdevs}")

def rescale(data:List[Vector])->List[Vector]:
    dim=len(data[0])
    means,stdevs=scale(data)

    rescaled=[v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i]>0:
                v[i]=(v[i]-means[i])/stdevs[i]
    return rescaled
means,stdevs=scale(rescale(vectors))
#print(f"The new means are:{means}")
#print(f"The new standard deviations are:{stdevs}")

#tqdm library genrates custom progress bars


def primes_up_to(n:int)->List[int]:
    primes=[2]

    with tqdm.trange(3,n) as t:
        for i in t:
            i_is_prime=not any(i%p==0 for p in primes)
            if i_is_prime:
                primes.append(i)
            t.set_description(f"{len(primes)}primes")
        
        return primes
my_primes=primes_up_to(100_000)

#Dimensionality Reduction
"""Principal Component Analysis(PCA), Dimesnionality Reduction is Mostly useful when your dataset has a large number of dimensions and you 
want to find a small subset that captures most of the variation"""

def de_mean(data:List[Vector])->List[Vector]:
    mean=vector_mean(data)
    return [subtract(vector,mean) for vector in data] 

def direction(w:Vector)->Vector: #makes the unit vector
    mag=magnitude(w)
    return[w_i/mag for w_i in w]
"""given a direction d(vector of magnitude 1),each row x in the matrix extends dot(x,d) in the d direction.And every nonzero vector 
w determines a direction if we rescale it to have magnitude 1:"""

def directional_variance(data:List[Vector],w:Vector)->float: #variance of data in the direction of vector
    w_dir=direction(w)
    return sum(dot(v,w_dir)**2 for v in data)

def directional_variance_gradient(data:List[Vector],w:Vector)->Vector:#gradient of the variance
    w_dir=direction(w)
    return[sum(2*dot(v,w_dir)*v[i] for v in data) for i in range(len(w))]

def first_principal_component(data:List[Vector],n:int=100,step_size:float=0.1)->Vector:
    guess=[1.0 for _ in data[0]]
    with tqdm.trange(n) as t:
        for _ in t:
            dv=directional_variance(data,guess)
            gradient=directional_variance_gradient(data,guess)
            guess=gradient_step(guess,gradient,step_size)#updates the guess by taking a step in the direction of the gradient 
            t.set_description(f"dv:{dv:.3f}")
    return direction(guess)

def project(v:Vector,w:Vector)->Vector:
    projection_length=dot(v,w)
    return scalar_multiply(projection_length,w)

def remove_projection_from_vector(v:Vector,w:Vector)->Vector:
    return subtract(v,project(v,w))

def remove_projection(data:List[Vector],w:Vector)->List[Vector]:
    return[remove_projection_from_vector(v,w) for v in data]

def pca(data:List[Vector],num_components:int)->List[Vector]:
    components:List[Vector]=[]
    for _ in range(num_components):
        component=first_principal_component(data)
        components.append(component)
        data=remove_projection(data,component)
    
    return components

def transform_vector(v:Vector,components:List[Vector])->Vector:
    return [dot(v,w) for w in components]

def transform(data:List[Vector],components:List[Vector])->List[Vector]:
    return [transform_vector(v,components) for v in data]

np.random.seed(0)
data = np.random.multivariate_normal([0, 0], [[3, 1], [1, 2]], 100)
data = [list(row) for row in data]

# Apply PCA to reduce the data to 2 principal components
num_components = 2
components = pca(data, num_components)
transformed_data = transform(data, components)

# Convert lists back to numpy arrays for plotting
data_np = np.array(data)
transformed_data_np = np.array(transformed_data)

# Plot original data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data')
plt.legend()

# Plot transformed data
plt.subplot(1, 2, 2)
plt.scatter(transformed_data_np[:, 0], transformed_data_np[:, 1], alpha=0.6, label='Transformed Data (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data after PCA')
plt.legend()

plt.tight_layout()
plt.show()


