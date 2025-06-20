import matplotlib.pyplot as plt
import math
import random
from collections import Counter
from typing import Tuple,List
import random

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z = -10.0
    hi_z = 10.0

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z

    return mid_z

def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    data = [binomial(n, p) for _ in range(num_points)]
    histogram = Counter(data)
    
    plt.bar([x - 0.4 for x in histogram.keys()], 
            [v / num_points for v in histogram.values()],
            0.8, color='0.75')
    
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_below(lo: float, mu: float = 0, sigma: float = 1) -> float:
    return normal_cdf(lo, mu, sigma)

def normal_upper_bound(p: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(1 - p, mu, sigma)

def normal_lower_bound(p: float, mu: float = 0, sigma: float = 1) -> float:
    return inverse_normal_cdf(p, mu, sigma)

def normal_two_sided_bounds(p: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    tail_probability = (1 - p) / 2
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)
    
extreme_value_count=0
for _ in range(1000):
    num_heads=sum(1 if random.random()<0.5 else 0 for _ in range(1000))
    if num_heads>=530 or num_heads<=470:
        extreme_value_count+=1
"""This loop counts how often the number of heads in 1000 flips of a fair
 coin (50% chance of heads) is either extremely high (530 or more)
   or extremely low (470 or fewer)."""



if __name__ == "__main__":
    binomial_histogram(0.75, 100, 10000)
    
    p = 0.95
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    upper_bound_95 = normal_upper_bound(p, mu_0, sigma_0)
    upper_p_value=normal_probability_above
    lower_p_value=normal_probability_below
    
    print(f"mu_0: {mu_0}, sigma_0: {sigma_0}")
    print(f"upper_bound_95: {upper_bound_95}")
    
    mu_0 = 500
    sigma_0 = 15.8
    result = two_sided_p_value(529.5, mu_0, sigma_0) #for p-values
    result1=two_sided_p_value(531.5,mu_0,sigma_0)
    result3=upper_p_value(524.5,mu_0,sigma_0)
    result4=upper_p_value(526.5,mu_0,sigma_0)
    result5=normal_two_sided_bounds(p,mu_0,sigma_0)
    
    print(f"two_sided_p_value: {result}") #p>5% therefore H0 is not rejected 

    print(f"two_sided_p_value;{result1}") #p<5% therefore H0 is rejected
    
    print(f"fdgw:{result3}") #p>5%

    print(f"gdcv:{result4}")#p<5%

    print(f"extreme_value:{extreme_value_count}")
    print(f"normal_two_sided_bound:{result5}")#gives the estimated interval

   
