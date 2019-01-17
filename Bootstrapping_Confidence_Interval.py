"""Using Bootstrapping method to get interval confidence"""
from random import choices
import numpy as np

resampling = 5000


def get_desire_statistics(input):
	#I take an example with mean
	return np.mean(input)

def get_confidence_interval_with_bootstrapping(input):
	statistics_collection = []
	for i in range(resampling):
		random_arr = choices(input, k=len(input))
		statistics_collection.append(get_desire_statistics(random_arr))
	return np.percentile(statistics_collection,2.5), \
		   np.percentile(statistics_collection,97.5)

mu, sigma = 0, 1
total_samples = 1000
#random_input = np.random.normal(mu, sigma, total_samples)
random_input = np.random.binomial(size=1000, n=1, p= 0.7)
print(np.mean(random_input))
print(get_confidence_interval_with_bootstrapping(random_input))
