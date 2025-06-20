"""Nearest Neighbour Model is one of the simples models, it takes inton account"""
#:Some Notion of distance
#:An assumption that points are close to one another and asre similiar

from typing import List,Tuple,NamedTuple,Dict
from collections import Counter,defaultdict
from vector import distance 
import requests
import csv
import matplotlib.pyplot as plt
import numpy as lp
Vector=List[float]
def raw_majority_vote(labels:List[str])->str:
    votes=Counter(labels)
    winner,_=votes.most_common(1)[0]

    return winner
print(f"the raw majority vote:{raw_majority_vote(['a','b','c','c'])}")

"""if there is a tie , the function returns the first element,to avoid it we have three options"""
#1:Pick one of the winners at random
#2:Weight the votes by distance and pick the weighted winner
#3:Reduce k until we find a unique winner
"""Lets proceed with the third"""
def majority_vote(labels:List[str])->str: #recusrsively reduces the list size until it finds a winner
    vote_counts=Counter(labels)
    winner,winner_count=vote_counts.most_common(1)[0]
    num_winners=len([count for count in vote_counts.values() if count==winner_count])

    if num_winners==1:
        return winner
    else:
        return majority_vote(labels[:-2])
    
print(f"The majority votes are from:{majority_vote(['a','b','c','b','a'])}")

class LabeledPoint(NamedTuple):
    point:Vector
    label:str

def knn_classify(k:int,labeled_points:List[LabeledPoint],new_point:Vector)->str:
    by_distance=sorted(labeled_points,key=lambda lp:distance(lp.point,new_point))

    k_nearest_labels=[lp.label for lp in by_distance[:k]]

    return majority_vote(k_nearest_labels)

data=requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
with open('iris.data','w') as f:
    f.write(data.text)
    

def parse_iris_row(row:List[str])->LabeledPoint:
    measurements=[float(value) for value in row[:-1]]
    label=row[-1].split("-")[-1]
    return LabeledPoint(measurements,label)

with open('iris.data') as f:
    reader=csv.reader(f)
    iris_data=[parse_iris_row(row) for row in reader]

points_by_species:Dict[str,List[Vector]]=defaultdict(list)
for iris in iris_data:
    points_by_species:Dict[str,List[Vector]]=defaultdict(list)

for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

metrics=['sepal length','sepal width','petal length','petal width']
pairs=[(i,j) for i in range(4) for j in range(4) if i<j]
marks=['+','.','x']
fig,ax=plt.subplots(2,3)
for row in range(2):
    for col in range(3):
        i,j=pairs[3*row+col]

        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}",fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        for mark,(species,points) in zip(marks,points_by_species.items()):
            xs=[point[i] for point in points]
            ys=[point[j] for point in points]
            ax[row][col].scatter(xs,ys,marker=mark,label=species)
    ax[-1][-1].legend(loc='lower right',prop={'size':6})
    plt.show()
