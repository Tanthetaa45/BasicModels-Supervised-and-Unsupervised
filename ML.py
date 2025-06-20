import random
from typing import TypeVar,List,Tuple

X=TypeVar('X')

def split_data(data:List[X],prob:float)->Tuple[List[X],List[X]]:
    data=data[:]
    random.shuffle(data)
    cut=int(len(data)*prob)
    return data[:cut],data[cut:]

data=[n for n in range(1000)]
train,test=split_data(data,0.75)

#print(f"the length of train is:{len(train)}")
#print(f"The length of the test:{len(test)}")
#print(f"vedgvg:{sorted(train+test)}")



Y = TypeVar('Y')



def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])

# Example SKM model class definition
class SKM:
    def __init__(self):
        self.model = None  # Placeholder for the model
    
    def train(self, x_train: List[X], y_train: List[Y]):
        # Placeholder training logic
        self.model = "trained_model"
        print("Training complete.")
    
    def test(self, x_test: List[X], y_test: List[Y]) -> float:
        # Placeholder testing logic, returning dummy performance
        print("Testing complete.")
        return 0.95  # Example performance metric

# Prepare the data
xs = [x for x in range(1000)]
ys = [2 * x for x in xs]

# Initial train-test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

print("Initial train-test split check:")
if all(y == 2 * x for x, y in zip(x_train, y_train)):
    print("All y_train values are 2 * x_train values.")
else:
    print("Not all y_train values are 2 * x_train values.")

if all(y == 2 * x for x, y in zip(x_test, y_test)):
    print("All y_test values are 2 * x_test values.")
else:
    print("Not all y_test values are 2 * x_test values.")

# Create and train the model
model = SKM()
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.33)
model.train(x_train, y_train)

# Test the model
performance = model.test(x_test, y_test)
print(f"Model performance: {performance}")
 

 