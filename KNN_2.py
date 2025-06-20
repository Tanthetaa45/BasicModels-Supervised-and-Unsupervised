from typing import List, NamedTuple, Dict,TypeVar,Tuple
from collections import Counter, defaultdict
from vector import distance  # Ensure you have a function to calculate distance
import requests
import csv
import matplotlib.pyplot as plt
import random
import tqdm

Vector = List[float]

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner



X=TypeVar('X')

def split_data(data:List[X],prob:float)->Tuple[List[X],List[X]]:
    data=data[:]
    random.shuffle(data)
    cut=int(len(data)*prob)
    return data[:cut],data[cut:]

print(f"The raw majority vote: {raw_majority_vote(['a', 'b', 'c', 'c'])}")

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # Use labels[:-1] to avoid reducing too fast

print(f"The majority votes are from: {majority_vote(['a', 'b', 'c', 'b', 'a'])}")

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    return majority_vote(k_nearest_labels)

data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
with open('iris.data', 'w') as f:
    f.write(data.text)

def parse_iris_row(row: List[str]) -> LabeledPoint:
    if len(row) < 5:  # Check if the row has enough elements
        raise ValueError("Row has fewer elements than expected.")
    measurements = [float(value) for value in row[:-1]]
    label = row[-1].split("-")[-1]
    return LabeledPoint(measurements, label)

iris_data = []
with open('iris.data') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:  # Skip empty rows
            try:
                iris_data.append(parse_iris_row(row))
            except ValueError as e:
                print(f"Skipping row: {row} due to error: {e}")

points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x']
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for row in range(2):
    for col in range(3):
        i, j = pairs[3*row + col]

        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)
ax[-1, -1].legend(loc='lower right', prop={'size': 6})
plt.show()

random.seed(12)
iris_train,iris_test=split_data(iris_data,0.70)
print(f"the length:{len(iris_train)}")
print(f"The length of test:{len(iris_test)}")

confusion_matrix:Dict[Tuple[str,str],int]=defaultdict(int)
num_correct=0

for iris in iris_test:
    predicted=knn_classify(5,iris_train,iris.point)
    actual=iris.label

    if predicted==actual:
        num_correct+=1

    confusion_matrix[(predicted,actual)]+=1
pct_correct=num_correct/len(iris_test)
print(pct_correct,confusion_matrix)

def random_point(dim:int)->Vector:
    return [random.random() for _ in  range(dim)]
def random_distances(dim:int,num_pairs:int)->List[float]:
    return [distance(random_point(dim),random_point(dim))
            for _ in range(num_pairs)]
dimensions=range(1,101)
avg_distances=[]
min_distances=[]
random.seed(0)

for dim in tqdm.tqdm(dimensions,desc="Curse of Dimensionality"):
    distances=random_distances(dim,10000)
    avg_distances.append(sum(distances)/10000)
    min_distances.append(min(distances))

    min_avg_ratio=[min_dist/avg_dist for min_dist,avg_dist in zip(
        min_distances,avg_distances)]
    

plt.figure(figsize=(18, 6))

# Plot for average distances
plt.subplot(1, 3, 1)
plt.plot(dimensions, avg_distances, label='Average Distance')
plt.xlabel('Dimension')
plt.ylabel('Average Distance')
plt.title('Average Distance vs Dimension')
plt.legend()

# Plot for minimum distances
plt.subplot(1, 3, 2)
plt.plot(dimensions, min_distances, label='Minimum Distance', color='red')
plt.xlabel('Dimension')
plt.ylabel('Minimum Distance')
plt.title('Minimum Distance vs Dimension')
plt.legend()

# Plot for the ratio of minimum distance to average distance
plt.subplot(1, 3, 3)
plt.plot(dimensions, min_avg_ratio, label='Min/Avg Distance Ratio', color='green')
plt.xlabel('Dimension')
plt.ylabel('Min/Avg Distance Ratio')
plt.title('Min/Avg Distance Ratio vs Dimension')
plt.legend()

plt.tight_layout()
plt.show()