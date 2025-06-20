#A generalization of the median is the quantile, which represents the value under which a certain percentile of the data lies (the median represents the value under which 50% of the data lies)
"""mean,median,mode,quantile,dispersion,variance,standard deviation & interquartile"""
import matplotlib.pyplot as plt

from collections import Counter
from typing import List,Tuple
import math
from vector import dot
def mean(xs:List[float])->float:
    return sum(xs)/len(xs)

num_friends=[100,56,78,65,98,46,76,84,65,87,78]
friends_counts = Counter(num_friends)
xs=range(11)
social_hours=[23,56,76,45,78,92,5,7,43,67,65]
ys = [friends_counts[x] for x in xs]
def quantile(xs:List[float],p:float)->float:
    p_index=int(p*len(xs))
    return sorted(xs)[p_index]
#assert quantile(num_friends,0.10)==56
#print("the 10th quantile is:" ,quantile(num_friends,0.10))
def mode(x:List[float])->List[float]:#returns a list since there might be more than one node
    counts=Counter(x)
    max_count=max(counts.values())
    return[x_i for x_i,count in counts.items() if count==max_count]
#print("the mode is:",set(mode(num_friends)))
def sum_of_squares(v):
    """Return the sum of the squares of the elements in v."""
    return sum(v_i ** 2 for v_i in v)

#dispersion refers to the measure of how spread out our data is,a very simple measure is the 'range'
def data_range(xs:List[float])->float:
    return max(xs)-min(xs)
#print(data_range(num_friends)) #100-54
def de_mean(xs:List[float])->List[float]:#Calculates the deviation of each element from the mean
    x_bar=mean(xs)
    return [x-x_bar for x in xs]
def variance(xs:List[float])->float:
    assert len(xs)>=2

    n=len(xs)
    deviations=de_mean(xs)
    return sum_of_squares(deviations)/(n-1) #sum of squared deviation /n-1 where n is the number of elements in the list

#print(variance(num_friends))

def standard_deviation(xs:List[float])->float:
    return math.sqrt(variance(xs))
#print(standard_deviation(num_friends))
#print(standard_deviation(social_hours))

def interquartile_range(xs:List[float])->float:
    return quantile(xs,0.75)-quantile(xs,0.25) #75th percentile-25th percentile
#print("The interquartile range is:",interquartile_range(num_friends))


def covariance(xs:List[float],ys:List[float])->float:
    assert len(xs)==len(ys)
    return dot(de_mean(xs),de_mean(ys))/(len(xs)-1)
#print("the covariance is:",covariance(num_friends,social_hours))

#correlation divides out the standard deviation of both variables:

def correlation(xs: List[float], ys: List[float]) -> float:
    """Calculates the Pearson correlation coefficient between two lists."""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        correlation_coefficient = covariance(xs, ys) / (stdev_x * stdev_y)
        #print("The Pearson correlation coefficient is:", correlation(num_friends,social_hours))
        return correlation_coefficient
    else:
       # print("Standard deviation of one or both variables is zero, correlation cannot be calculated.",correlation)
        return 0

    

