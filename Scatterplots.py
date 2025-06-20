import matplotlib.pyplot as plt
friends=[70,65,72,63,71,64,60,64,67]
minutes=[175,170,205,120,220,130,105,145,190]
labels=['a','b','c','d','e','f','g','h','i']

plt.scatter(friends,minutes)

for label,friends_count,minute_count in zip(labels,friends,minutes):
    plt.annotate(label,
        xy=(friends_count,minute_count),
        xytext=(5,-5),
        textcoords='offset points')
    
plt.title("daily minutes vs number of friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()