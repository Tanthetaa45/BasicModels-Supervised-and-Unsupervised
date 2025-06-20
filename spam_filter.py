"""We can calculate that a message is spam or not buy
using the formula"""
"""P(S|X=x)=P(X=x|S)/[P(X=x|S)+P(X=x|not S)]"""
"""avoid multiplying lots of probabilities together,
to prevent UNDERFLOW,as computers dont deal well with floating
point numbers too close to 0"""

"""Choose a pseudocount_k_ and estimate the probablity of 
seeing the ith word in a spam message as 

P(Xi|S)=(k+number of spams containing wi)/(2k+ number of spams)
P(Xi|not S)=(k+number of non spams containing wi)/(2k+number of non spams)"""

from typing import Set,NamedTuple,List,Tuple,Dict,Iterable,TypeVar
import math,random
from collections import defaultdict,Counter
import re,glob
from io import BytesIO
import requests 
import tarfile

X=TypeVar('X')
def split_data(data:List[X],prob:float)->Tuple[List[X],List[X]]:
    data=data[:]
    random.shuffle(data)
    cut=int(len(data)*prob)
    return data[:cut],data[cut:]

def tokenize(text:str)->Set[str]:
    text=text.lower()
    all_words=re.findall("[a-z0-9']+",text)
    return set(all_words)
print(f"The individuals are:{tokenize('Data Science is science')}")

class Message(NamedTuple):
    text:str
    is_spam:bool

#nonspam emails as ham emails
class NaiveBayesClassifier:
    def __init__(self,k:float=0.5)->None:
        self.k=k #k is the smoothing factor 
        
        self.tokens:Set[str]=set()
        self.token_spam_counts:Dict[str,int]=defaultdict(int)
        self.token_ham_counts:Dict[str,int]=defaultdict(int)
        self.spam_messages=self.ham_messages=0

    def train(self,messages:Iterable[Message])->None:
        for message in messages:
            if message.is_spam:
                self.spam_messages+=1
            else:
                self.ham_messages+=1
            
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token]+=1
                else:
                    self.token_ham_counts[token]+=1
    def _probabilities(self,token:str)->Tuple[float,float]:
     """returns P(token|spam) and P(token|ham)"""
     spam=self.token_spam_counts[token]
     ham=self.token_ham_counts[token]

     p_token_spam=(spam+self.k)/(self.spam_messages+2*self.k)
     p_token_ham=(ham+self.k)/(self.ham_messages+2*self.k)

     return p_token_spam,p_token_ham
    

    def predict(self,text:str)->float:
        text_tokens=tokenize(text)
        log_prob_if_spam=log_prob_if_ham=0.0

        for token in self.tokens:
            prob_if_spam,prob_if_ham=self._probabilities(token)

            if token in text_tokens:
                log_prob_if_spam+=math.log(prob_if_spam)
                log_prob_if_ham+=math.log(prob_if_ham)
            else:
                log_prob_if_spam+=math.log(1.0-prob_if_spam)
                log_prob_if_ham+=math.log(1.0-prob_if_ham)
        prob_if_spam=math.exp(log_prob_if_spam)
        prob_if_ham=math.exp(log_prob_if_ham)
        return prob_if_spam/(prob_if_spam+prob_if_ham)
    

messages=[Message("spam rules",is_spam=True),Message("ham rules",is_spam=False),
          Message("hello ham",is_spam=False)]

model=NaiveBayesClassifier(k=0.5)
model.train(messages)

print(f"The counts are:{model.tokens}")

text="hello spam"



# Define the probability lists
probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),  # "spam" (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham" (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)  # "hello" (present)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),  # "spam" (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham" (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5)  # "hello" (present)
]

# Calculate p_if_spam and p_if_ham
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Calculate prediction
prediction = p_if_spam / (p_if_spam + p_if_ham)

# Print the prediction value
print(f"Prediction: {prediction:.2f}")



BASE_URL="https://spamassassin.apache.org/old/publiccorpus"
FILES=["20021010_easy_ham.tar.bz2","20021010_hard_ham.tar.bz2","20021010_spam.tar.bz2"]

OUTPUT_DIR='spam_data'
for filename in FILES:
    content=requests.get(f"{BASE_URL}/{filename}").content
    fin=BytesIO(content)

    with tarfile.open(fileobj=fin,mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)


path='spam_data/*/*'
data:List[Message]=[]
for filename in glob.glob(path):
    is_spam="ham" not in filename

    with open(filename,errors='ignore')as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject=line.lstrip("Subject:")
                data.append(Message(subject,is_spam))
                break


random.seed(0)
train_messages,test_messages=split_data(data,0.75)
model=NaiveBayesClassifier()
model.train(train_messages)

predictions=[(message,model.predict(message.text)) for message in 
             test_messages]

confusion_matrix=Counter((message.is_spam,spam_probability>0.5) for 
                         message ,spam_probability in predictions)

print(confusion_matrix)


def p_spam_given_token(token:str,model:NaiveBayesClassifier)->float:
    prob_if_spam,prob_if_ham=model._probabilities(token)
    return prob_if_spam/(prob_if_spam+prob_if_ham)

words=sorted(model.tokens,key=lambda t:p_spam_given_token(t,model))

print("spammiest_words",words[-10:])
print("hammiest_words",words[:10])


def drop_final_s(word):
    return re.sub("s$","",word)
word="apples"
result=drop_final_s(word)
print(f"fgfmhj:{result}")


