"""Using Bootstrapping method to compute standard error of the mean"""
def get_std_error_without_bootstrap(random_output):
	sample_mean = random_output.mean()
	var = 0
	for i in range(len(random_output)):
		var  = (random_output[i] - sample_mean) ** 2
	var /= (len(random_output) - 1)
	pop_stdev = var**0.5/(len(random_output)**0.5)
	print("get_std_error_without_bootstrap")
	print(pop_stdev)


def get_std_error_with_bootstrap(random_output):
	list_of_means = []
	for i in range(100000):
		list_of_means.append(np.mean(choices(random_output, k=len(random_output))))
	sample_mean = sum(list_of_means)/(len(list_of_means))
	var = 0
	for i in range(len(list_of_means)):
		var  = (list_of_means[i] - sample_mean)**2
	var /= (len(list_of_means) - 1)
	pop_stdev = var**0.5
	print("get_std_error_with_bootstrap")
	print(pop_stdev)

from scipy import stats
from random import choices
import math
np.random.seed(10)
mu, sigma = 1, 2
total_samples = 20
random_output = np.random.normal(mu, sigma, total_samples)
print(random_output)
print(len(random_output))
print(get_std_error_without_bootstrap(random_output))
print(get_std_error_with_bootstrap(random_output))
print("ground_truth")
print(5/(total_samples**0.5))
