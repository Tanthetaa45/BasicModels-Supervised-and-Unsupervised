import matplotlib.pyplot as plt

from collections import Counter

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Count grades in deciles (0-90, 10-90, ..., 90-100)
histogram = Counter(grade // 10 * 10 for grade in grades)

# Create the bar chart
plt.bar([x + 5 for x in histogram.keys()], histogram.values(), 10, edgecolor=(0, 0, 0))  

#This line uses a list comprehension to create a Counter object named histogram.
#The list comprehension iterates through each grade in the grades list.
#Inside the comprehension, grade // 10 * 10 calculates the decile (0-90, 10-90, ..., 90-100) for each grade by dividing by 10 and then multiplying by 10 to get the nearest multiple of 10.
#The Counter object then counts the occurrences of each decile value.

# Set axis limits and labels
plt.axis([-5, 105, 0, max(histogram.values()) + 1])  # Adjust upper limit for clarity
plt.xticks([10 * i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")

# Display the chart
plt.show()
