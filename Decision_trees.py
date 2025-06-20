import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from typing import List,Tuple,TypeVar,Any,NamedTuple,Optional,Dict,Union
import random,math
from collections import Counter,defaultdict
#import tqdm
#from Rescaling import rescale
#from MLR import least_squares_fit,predict
#from GradientDescent import gradient_step
Vector=List[float]

"""Decision Trees can easily handle a mix of numeric and categorical attributes and can even classify data for which attributes
 are missing"""
"""Decision trees can be categorised into classification trees(give categorical Outputs) and regression trees(give
numerical outputs)"""
"""Entropy here is associated with the amount of uncertainity in data"""

def entropy(class_probabilities:List[float])->float:
    return sum(-p*math.log(p,2) for p in class_probabilities if p>0)

print(f"The entropy is:{entropy([0.25,0.75])}")

def class_probabilities(labels:List[Any])->List[float]:
    total_count=len(labels)
    return [count/total_count for count in Counter(labels).values()]

def data_entropy(labels:List[Any])->float:
    return entropy(class_probabilities(labels))

print(f"The Data entropy is :{data_entropy([True,False])}")


def partition_entropy(subsets:List[List[Any]])->float:
    total_count=sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset)*len(subset)/total_count for subset in subsets)

class Candidate(NamedTuple):
    level:str
    lang:str
    tweets:bool
    phd:bool
    did_well:Optional[bool]=None

inputs=[Candidate('Senior','Java',False,False,False),
        Candidate('Senior','java',False,True,False),
        Candidate('Mid','Python',False,False,True),
        Candidate('Junior','Python',False,False,True),
        Candidate('Junior','R',True,False,True),
        Candidate('Junior','R',True,True,False),
        Candidate('Mid','R',True,True,True),
        Candidate('Senior','Python',False,False,False),
        Candidate('Senior','R',True,False,True),
        Candidate('Junior','Python',True,False,True),
        Candidate('Senior','Python',True,True,True),
        Candidate('Mid','Python',False,True,True),
        Candidate('Mid','Java',True,False,False),
        Candidate('Junior','Python',False,True,False)]
"""Decison Nodes Ask a question and direct us differently depending upon the answer"""
"""Leaf nodes Give us a prediction"""

T=TypeVar('T')
def partition_by(inputs:List[T],attribute:str)->Dict[Any,List[T]]:
    partitions:Dict[Any,List[T]]=defaultdict(list)
    for input in inputs:
        key=getattr(input,attribute)
        partitions[key].append(input)
    return partitions

def partition_entropy_by(inputs:List[Any],attribute:str,label_attribute:str)->float:
    partitions=partition_by(inputs,attribute)
    labels=[[getattr(input,label_attribute)for input in partition] for partition in 
            partitions.values()]
    return partition_entropy(labels)

for key in ['level','lang','tweets','phd']:
    print(key,partition_entropy_by(inputs,key,'did_well'))

print(f"P_entropy:{partition_entropy_by(inputs,'lang','did_well')}") #lower the entroppy value, higher is the data fit for the model

senior_inputs=[input for input in inputs if input.level=='Senior']

print(f"The PArtition_entropy is:{partition_entropy_by(senior_inputs,'phd','did_well')}")

class Leaf(NamedTuple):
    value:Any
class Split(NamedTuple):
    attribute:str
    subtrees:dict
    default_value:Any=None

Decison_Tree=Union[Leaf,Split]

def classify(tree:Decison_Tree,input:Any)->Any:
    if isinstance(tree,Leaf): #returns the value if the tree node is a leaf
        return tree.value
    """Otherwise this tree consists of an attribute to split on and a dictionary whose keys 
    are values of that attribute and whose values are subtress to consider next"""
    subtree_key=getattr(input,tree.attribute)

    if subtree_key not in tree.subtrees: #if no subtree for a key return the default value
        return tree.default_value
    subtree=tree.subtrees[subtree_key] #choose the apppropriate subtree and use it to classify theinput
    
    return classify(subtree,input)

def build_tree_id3(inputs:List[Any],split_attributes:List[str],target_attribute:str)->Decison_Tree:
    label_counts=Counter(getattr(input,target_attribute) for input in inputs)
    most_common_label=label_counts.most_common(1)[0][0]
    if len(label_counts)==1:
        return Leaf(most_common_label)
    if not split_attributes:
        return Leaf(most_common_label)
    
    def split_entropy(attribute:str)->float:
        return partition_entropy_by(inputs,attribute,target_attribute)
    best_attribute=min(split_attributes,key=split_entropy)
    partitions=partition_by(inputs,best_attribute)
    new_attributes=[a for a in split_attributes if a!=best_attribute]

    subtrees={attribute_value:build_tree_id3(subset,new_attributes,target_attribute)for attribute_value
              ,subset in partitions.items()}
    return Split(best_attribute,subtrees,default_value=most_common_label)


tree=build_tree_id3(inputs,['level','lang','tweets','phd'],'did_well')

print(f'The value is: {classify(tree, Candidate("Intern", "Java", True, True))}')

"""Random Forests builds multiple decision trees and combines their outputs. If they are Classific
ation trees , we let them vote and if they are regression trees , we average their predictions"""


if len(split_candidates)<=self.num_split_candidates:
    sampled_split_candidates=split_candidates
else:
    sampled_split_candidates=random.sample(split_candidates,self.num_split_candidates)

best_attribute=min(sampled_split_candidates,key=split_entropy)

partitions=partition_by(inputs,best_attribute)
