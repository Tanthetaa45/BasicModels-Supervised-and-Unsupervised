import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
import math

class Statistics:
    def __init__(self, data: List[float]):
        self.data = data

    @staticmethod
    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    @staticmethod
    def quantile(xs: List[float], p: float) -> float:
        p_index = int(p * len(xs))
        return sorted(xs)[p_index]

    @staticmethod
    def mode(xs: List[float]) -> List[float]:
        counts = Counter(xs)
        max_count = max(counts.values())
        return [x_i for x_i, count in counts.items() if count == max_count]

    @staticmethod
    def sum_of_squares(v: List[float]) -> float:
        return sum(v_i ** 2 for v_i in v)

    @staticmethod
    def data_range(xs: List[float]) -> float:
        return max(xs) - min(xs)

    @staticmethod
    def de_mean(xs: List[float]) -> List[float]:
        x_bar = Statistics.mean(xs)
        return [x - x_bar for x in xs]

    @staticmethod
    def variance(xs: List[float]) -> float:
        assert len(xs) >= 2
        n = len(xs)
        deviations = Statistics.de_mean(xs)
        return Statistics.sum_of_squares(deviations) / (n - 1)

    @staticmethod
    def standard_deviation(xs: List[float]) -> float:
        return math.sqrt(Statistics.variance(xs))

    @staticmethod
    def interquartile_range(xs: List[float]) -> float:
        return Statistics.quantile(xs, 0.75) - Statistics.quantile(xs, 0.25)

    @staticmethod
    def covariance(xs: List[float], ys: List[float]) -> float:
        assert len(xs) == len(ys)
        return sum(x * y for x, y in zip(Statistics.de_mean(xs), Statistics.de_mean(ys))) / (len(xs) - 1)

    @staticmethod
    def correlation(xs: List[float], ys: List[float]) -> float:
        stdev_x = Statistics.standard_deviation(xs)
        stdev_y = Statistics.standard_deviation(ys)
        if stdev_x > 0 and stdev_y > 0:
            return Statistics.covariance(xs, ys) / (stdev_x * stdev_y)
        else:
            return 0


# Example usage
num_friends = [100, 56, 78, 65, 98, 46, 76, 84, 65, 87, 78]
social_hours = [23, 56, 76, 45, 78, 92, 5, 7, 43, 67, 65]

stats = Statistics(num_friends)

print("The 10th quantile is:", Statistics.quantile(num_friends, 0.10))
print("The mode is:", set(Statistics.mode(num_friends)))
print("Data range:", Statistics.data_range(num_friends))
print("Variance:", Statistics.variance(num_friends))
print("Standard deviation:", Statistics.standard_deviation(num_friends))
print("Standard deviation (social hours):", Statistics.standard_deviation(social_hours))
print("Interquartile range:", Statistics.interquartile_range(num_friends))
print("Covariance:", Statistics.covariance(num_friends, social_hours))
print("Correlation:", Statistics.correlation(num_friends, social_hours))
