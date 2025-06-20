import matplotlib.pyplot as plt
from collections import Counter
from typing import List,Tuple

# Define num_friends with 50 additional data points
num_friends = [
    100, 49, 41, 40, 25, 19, 19, 18, 18, 18, 16, 15, 15, 15, 14, 14, 13, 13, 13, 12,
    12, 12, 11, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

# Calculate the histogram
friends_counts = Counter(num_friends)
xs = range(101)
ys = [friends_counts[x] for x in xs]

# Plot the histogram
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
#plt.show()

# Calculate and print required values
num_points = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)
sorted_values = sorted(num_friends)

print(f"Number of points: {num_points}")
print(f"Largest value: {largest_value}")
print(f"Smallest value: {smallest_value}")
print(f"Sorted values: {sorted_values}")

def mean(xs:List[float])->float:
    return sum(xs)/len(xs)
mean=mean(num_friends)
print(f"Mean value:{mean}")
