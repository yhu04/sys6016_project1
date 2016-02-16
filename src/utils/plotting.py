

import matplotlib.pyplot as plt

def plot_dataframe_attribute(dataframe, attribute, options=[]):

	num_elems = len(dataframe[attribute])

	print dataframe[attribute].tolist()

	plt.figure()
	plt.plot(range(num_elems), dataframe[attribute].tolist(), 'bo')
	if "mean" in options:
		mean = dataframe[attribute].mean()
		print "mean: " + str(mean)
		plt.plot(range(num_elems), [mean] * num_elems)
	if "median" in options:
		median = dataframe[attribute].median()
		print "median: " + str(median)
		plt.plot(range(num_elems), [median] * num_elems)
	if "mode" in options:
		mode = dataframe[attribute].mode()
		print "mode: " + str(mode)
		plt.plot(range(num_elems), [mode] * num_elems)
	if "std" in options:
		std = dataframe[attribute].std()
		print "std: " + str(std)
		plt.plot(range(num_elems), [dataframe[attribute].mean() + std] * num_elems)
		plt.plot(range(num_elems), [dataframe[attribute].mean() - std] * num_elems)		
	plt.show()

