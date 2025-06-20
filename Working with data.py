from statistics import correlation
from typing import List, Dict,NamedTuple,Optional
from collections import Counter,namedtuple,defaultdict
import math,re
import matplotlib.pyplot as plt
import random
import matplotlib.image as mping 
from Statistics1 import Statistics
from Matrix import make_matrix
from dataclasses import dataclass
import datetime



Vector=List[float]
Matrix=List[List[float]]


def bucketize_point(point: float, bucket_size: float) -> float:
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    return Counter(bucketize_point(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)

random.seed(0)

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    # If not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z = -10.0
    hi_z = 10.0

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        # print(f"mid_z:{mid_z},mid_p:{mid_p}")
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z

    return mid_z

uniform = [200 * random.random() - 100 for _ in range(10000)]
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")
plt.show()

def random_normal()->float:
    return inverse_normal_cdf(random.random())
xs=[random_normal() for _ in range(1000)]
ys1=[x+ random_normal()/2 for x in xs]
ys2=[-x+random_normal()/2 for x in xs]

plt.scatter(xs,ys1,marker='.',color='green',label='ys1')
plt.scatter(xs,ys2,marker='.',color='yellow',label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Different Joint Distributions Of the Above Histogram")
plt.show()
print(Statistics.correlation(xs,ys1)) #0.9
print(Statistics.correlation(xs,ys2)) #-0.89
"""For Multiple Dimensions"""
def correlation_matrix(data: List[Vector]) -> Matrix:
    
    # Define a function to calculate correlations
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])
    
    # Use make_matrix to create the correlation matrix
    return make_matrix(len(data),len(data), correlation_ij)
#corr_data is a list of 4 100d vectors
corr_data = []
num_vectors = 4
vector_dimension = 100

# Populate corr_data with random values
for _ in range(num_vectors):
    vector = [random.random() for _ in range(vector_dimension)]
    corr_data.append(vector)

fig,ax=plt.subplots(num_vectors,num_vectors)
for i in range (num_vectors):
   for j in range(num_vectors): 
       if i!=j:ax[i][j].scatter(corr_data[j],corr_data[i])
       else:ax[i][j].annotate("series"+str(i),(0.5,0.5),xycoords='axes fraction',ha="center",
                              va="center")
       if i<num_vectors-1:ax[i][j].xaxis.set_visible(False)
       if j>0:ax[i][j].yaxis.set_visible(False)
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

"""NamedTuple is like a tuple but with named slots and it solves the type 
Annotation issue with namedtuples and using dicts"""



class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

# Create an instance of StockPrice
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)
# Call the method is_high_tech
print(price.is_high_tech())  # This will print: True

"""NamedTuples are immutable whereas DataClasses are mutable therefore 
its a wiser and better option to use DataClasses""" 
"""Syntax is sismilar to NamedTuples but for inheritance we use 
a decorator"""




@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 108.96)


print(price2.closing_price)  #  print: 108.96


print(price2.is_high_tech())  # print: True

price2.closing_price/=2
print(price2.closing_price) 
"""the instance variables in dataclass can be modified ,the feature objects in namedtuples lack"""

"""we can also parse the function"""
from dateutil.parser import parse


def parse_row(row:List[str])->StockPrice:
    symbol,date,closing_price=row
    return StockPrice(symbol=symbol,date=parse(date).date(),closing_price=float(closing_price))

stock=parse_row(["MSFT","2018-12-14","106.03"])


assert stock.date==datetime.date(2018,12,14)

def try_parse_row(row:List[str])->Optional[StockPrice]:
    symbol,date_,closing_price_=row

    if not re.match(r"^[A-Z]+$",symbol):
        return None
    try:
        date=parse(date_).date()
    except ValueError:
        return None
    try:
        closing_price=float(closing_price_)
    except ValueError:
        return None
    return StockPrice(symbol,date,closing_price)

parsed_row= try_parse_row(["MSFT","2018-12--14","106.03"])
#assert parsed_row is not None, "Parsing failed"
#print(f"Parsing succeeded: {parsed_row}") #returns an error 


"""to know the highest ever closing price for each stock in our dataset"""
"""We use a default dict"""
from datetime import date


data = [
    StockPrice(symbol='MSFT', date=date(2018, 12, 24), closing_price=106.03),
    StockPrice(symbol='MSFT', date=date(2018, 12, 25), closing_price=107.03),
    StockPrice(symbol='AAPL', date=date(2014, 6, 20), closing_price=90.91),
    StockPrice(symbol='AAPL', date=date(2014, 6, 21), closing_price=91.91),
    StockPrice(symbol='GOOGL', date=date(2019, 7, 3), closing_price=1125.61),
    StockPrice(symbol='GOOGL', date=date(2019, 7, 4), closing_price=1126.61),
    StockPrice(symbol='AMZN', date=date(2020, 5, 14), closing_price=2409.78),
    StockPrice(symbol='AMZN', date=date(2020, 5, 15), closing_price=2410.78),
    StockPrice(symbol='FB', date=date(2021, 11, 1), closing_price=324.17),
    StockPrice(symbol='FB', date=date(2021, 11, 2), closing_price=325.17),
    StockPrice(symbol='TSLA', date=date(2022, 8, 5), closing_price=869.74),
    StockPrice(symbol='TSLA', date=date(2022, 8, 6), closing_price=870.74)
]

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

prices = {symbol: sorted(symbol_prices, key=lambda sp: sp.date) for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return (today.closing_price / yesterday.closing_price) - 1

class DailyChange(NamedTuple):
    symbol: str
    date: date
    pct_change: float

def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    return [
        DailyChange(symbol=today.symbol, date=today.date, pct_change=pct_change(yesterday, today))
        for yesterday, today in zip(prices, prices[1:])
    ]

all_changes = [
    change for symbol_prices in prices.values() for change in day_over_day_changes(symbol_prices)
]

# Print all changes for debugging
print("All Changes:")
for change in all_changes:
    print(change)

max_change = max(all_changes, key=lambda change: change.pct_change)
min_change = min(all_changes, key=lambda change: change.pct_change)

# Print statements for the assert conditions
print(f"Max Change: {max_change}")
print(f"Max Change pct_change: {max_change.pct_change}")
print(f"Min Change: {min_change}")
print(f"Min Change pct_change: {min_change.pct_change}")

# Adjust the assertions based on the actual data
assert max_change.symbol == 'AAPL', f"Expected 'AAPL', but got {max_change.symbol}"
assert max_change.date == date(2014, 6, 21), f"Expected {date(2014, 6, 21)}, but got {max_change.date}"
assert 0.01 < max_change.pct_change < 0.02, f"Expected percentage change between 0.01 and 0.02, but got {max_change.pct_change}"

assert min_change.symbol == 'AAPL', f"Expected 'AAPL', but got {min_change.symbol}"
assert min_change.date == date(2014, 6, 21), f"Expected {date(2014, 6, 21)}, but got {min_change.date}"
assert 0.01 < min_change.pct_change < 0.02, f"Expected percentage change between 0.01 and 0.02, but got {min_change.pct_change}"

changes_by_month = {month: [] for month in range(1, 13)}
for change in all_changes:
    changes_by_month[change.date.month].append(change)

avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
    for month, changes in changes_by_month.items()
}

avg_daily_change[10] = max(avg_daily_change.values())
print(f"Average Change: {avg_daily_change[0]}")