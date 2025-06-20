import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pn
import math
import enum,random
from collections import Counter
from typing import List,Tuple






"""probability density function(PDF),probability of seeing a value in a certain interval equals the
integral of the density function over the interval"""
class ProbabilityDistribution:
 def uniform_pdf(x:float)->float:
    return 1 if 0<=x<1 else 0
 def uniform_cdf(x:float)->float: #Cumulative distribution function gives the probability that the value is less than or equal to a certain value
    if x<0:return 0 #never less than zero
    elif x<1:return x
    else: return 1 #uniform random is always less than 1


#NORMAL DISTRIBUTION FUNCTION 
SQRT_TWO_PI=math.sqrt(2*math.pi)
def normal_pdf(x:float,mu:float=0,sigma:float=1)->float:
    return (math.exp(-(x-mu)**2/2/sigma**2)/SQRT_TWO_PI*sigma)
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1)for x in xs],'-',label='mu=0,sigma=1') #standard normal distribution
plt.plot(xs,[normal_pdf(x,sigma=2)for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5)for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1)for x in xs],'|',label='mu=-1,sigma=1')
plt.legend()
plt.title("various Normal pdfs")
plt.show()

def normal_cdf(x:float,mu:float=0,sigma:float=1)->float:
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma))/2
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x,sigma=1)for x in xs],'-',label='mu=0,sigma=1') #standard normal distribution
plt.plot(xs,[normal_cdf(x,sigma=2)for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5)for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1)for x in xs],'|',label='mu=-1,sigma=1')
plt.legend(loc=4)
plt.title("various Normal cdfs")
plt.show()

#inversion of normal cdfs give the corresponding values to a specified probablity and to find the inverse we use a binary search
"""Finding approximiate inverse using binary search"""

def inverse_normal_cdf(p:float,mu:float=0,sigma:float=1,tolerance:float=0.00001)->float:

#if not standard,compute standard and rescale
 if mu!=0 or sigma !=1:
    return mu+sigma*inverse_normal_cdf(p,tolerance=tolerance)
 low_z=-10.0
 hi_z=10.0

 while hi_z-low_z>tolerance:
    mid_z=(low_z+hi_z)/2
    mid_p=normal_cdf(mid_z)
    print(f"mid_z:{mid_z},mid_p:{mid_p}")
    if mid_p<p:
       low_z=mid_z
    else:
       hi_z=mid_z

 return mid_z

p=0.95
result=inverse_normal_cdf(p)
print(f"result for p={p}:{result}")

"""def bernoulli_trial(p:float)->int:
   return 1 if random.random()<p else 0
def binomial(n:int,p:float)->int:
   return sum(bernoulli_trial(p) for _ in range(n))
def binomial_histogram(p:float,n:int,num_points:int)->None:
   data=[binomial(n,p)for _ in range(num_points)]
   print(data[:100])

   histogram=Counter(data)
   print(histogram)
   plt.bar([x-0.4 for x in histogram.keys()],
           [v/num_points for v in histogram.values()],
           0.8,color='0.75')
   mu=p*n
   sigma=math.sqrt(n*p*(1-p))
   xs=range(min(data),max(data)+1)
   ys=[normal_cdf(i+0.5,mu,sigma)-normal_cdf(i-0.5,mu,sigma) for i in xs]
   plt.plot(xs,ys)
   plt.title("Binomial Distribution vs. Normal Approximation")
   plt.show()
   binomial_histogram(0.75,100,10000)""" #in probability3
                       
                       