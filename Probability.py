import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pn
import math
import enum,random
from collections import Counter
from typing import List,Tuple

#P(E,F)=P(E)P(F) for independent events
#P(E,F)=P(E/F)P(F) for dependent events

class Kid(enum.Enum):
    BOY=0
    GIRL=1

num_trials = int(input("Enter the number of trials: ")) #takes input of the number of trials


def random_kid()->Kid:
    return random.choice([Kid.BOY,Kid.GIRL])
both_girls=0
older_girl=0
either_girl=0

random.seed(0)
for _ in range(num_trials):
    younger=random_kid()
    older=random_kid()
    if older==Kid.GIRL:
        older_girl+=1
    if older==Kid.GIRL and younger==Kid.GIRL:
        both_girls+=1
    if older==Kid.GIRL or younger==Kid.GIRL:
        either_girl+=1

print("P(both|older):",both_girls/older_girl if older_girl!=0 else 0)
print("P(both|either):",both_girls/either_girl if either_girl!=0 else 0)


