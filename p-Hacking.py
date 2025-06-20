from tracemalloc import Statistic
import matplotlib.pyplot as plt
import math
import random
from collections import Counter
from typing import Tuple,List
import random

"""p-Hacking"""
#a procedure that erronously rejects the null hypothesis of 5% 

def run_experiment()->List[bool]:
    """True=Heads,False=Tails"""
    return [random.random()<0.5 for _ in range(1000)] #flips coin 1000 times

def reject_fairness(experiment:List[bool])->bool:
    """Uses the 5% significance rules"""
    num_heads=len([flip for flip in experiment if flip])
    return num_heads<469 or num_heads>531
random.seed(0)
experiments=[run_experiment() for _ in range(1000)]
num_rejections=len([experiment for experiment in experiments if reject_fairness(experiment)])

result=num_rejections
print(f"the number of rejections are:{result}")

"""A/B test"""
def estimated_parameters(N:int,n:int)->Tuple[float,float]:
    p=n/N
    sigma=math.sqrt(p*(1-p)/N)
    return p,sigma #pB-pA=math.sqrt(sigmaA**2+sigmaB**2)

def a_b_test_statistic(N_A:int,n_A:int,N_B:int,n_B:int)->float:
    p_A,sigma_A=estimated_parameters(N_A,n_A)
    p_B,sigma_B=estimated_parameters(N_B,n_B)
    return(p_B-p_A)/math.sqrt(sigma_A**2+sigma_B**2)

z=a_b_test_statistic(1000,200,1000,180)
z2=a_b_test_statistic(1000,200,1000,150)
print(f"z is:{z},{z2}")
