"""True Positive: is spam and predicted Spam
False Positive(Type 1 error): is not Spam but predicted Spam
False Negative(Type 2 error): is Spam but predicted not Spam
True Negative:is not Spam and predicted not Spam"""


"""We have to predict whether a boy names Luke has Leukamia or not:
In the Us alone every 5 out 0f 1000 babies are named Luke and the lifetime 
prevalance of leukamia is 1.4%.i.e 14 out of every 1000 persons"""

def accuracy(tp:int,fp:int,fn:int,tn:int)->float:
    correct=tp+tn
    total=tp+tn+fp+fn
    return correct/total
print(f"Accuracy is:{accuracy(70,4930,13930,981070)}")

def precision(tp:int,fp:int,fn:int,tn:int)->float:
    return tp/(tp+fp)
print(f"The precison is:{precision(70,4930,13930,981070)}")

"""recall measures what fraction of the positives our model identified"""
 
def recall(tp:int,fp:int,fn:int,tn:int)->float:
    return tp/(tp+fn)
print(f"recall value:{recall(70,4930,13930,981070)}")
"""precision and recall are combined into the F1 score"""
def f1_score(tp:int,fp:int,fn:int,tn:int)->float:

    p=precision(tp,fp,fn,tn)
    r=recall(tp,fp,fn,tn)
    return 2*p*r/(p+r) #the harmonic mean of precision and recall

"""The Bias-Variance Tradeoff"""

#high bias and low variance:underfitting
#low bias and high variance:overfitting