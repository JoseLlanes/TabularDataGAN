from collections import Counter
import numpy as np


class CategoricalToNumericalNorm:


    @staticmethod
    def create_category_intervals(data):
        # Count frequency of each unique category
        freq_counts = Counter(data)
        total = len(data)

        freq_ratios = {key: count / total for key, count in sorted(freq_counts.items())}

        intervals = {}
        start = 0.0
        for category, ratio in freq_ratios.items():
            end = start + ratio
            intervals[category] = np.array([start, end])
            start = end  # Update start for next category

        return intervals


    @staticmethod
    def truncated_normal_distribution(a, b):
        mu = (a + b) / 2
        sigma = (b - a) / 6

        while True:
            sample = np.random.normal(mu, sigma)
            if a <= sample <= b:
                return sample


    @classmethod
    def generate_number(cls, category, interval_category):
        return cls.truncated_normal_distribution(*interval_category[category])


    @staticmethod
    def inverse_categorical_interval(number, interval_category):
        for category, interval in interval_category.items():
            if interval[0] <= number <= interval[1]:
                return category
