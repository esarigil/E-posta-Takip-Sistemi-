
import random
from __data_preparation.utils import *

class Splitter:
	"""Veri setini boler. Eğitim ve test"""
	def __init__(self, params):
		self.split(params)
	def split(self, params):
		data = read_csv(params['datadump'], params['delimiter'])
		train = csv.writer(open(params['traindump'], 'w', newline=''),
			delimiter=params['delimiter'])
		test = csv.writer(open(params['testdump'], 'w', newline=''),
			delimiter=params['delimiter'])

		# Write header
		train.writerow(data[0])
		test.writerow(data[0])

		# Shuffle rest
		data = data[1:]
		random.shuffle(data)

		# Split data based on factor
		f = params['train_data_split_factor']
		train_data = data[:int((len(data)+1) * f)]
		test_data = data[int(len(data)* f + 1):]

		# Write data to csv files
		[train.writerow(row) for row in train_data]
		[test.writerow(row) for row in test_data]

		print("Original dump length:", len(data))
		print("Written", len(train_data), "rows to \"" + params['traindump']
			+ "\" and", len(test_data), "rows to \"" + params['testdump']
			+ "\" used a factor of:", f)
