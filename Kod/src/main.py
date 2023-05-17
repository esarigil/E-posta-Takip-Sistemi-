#!/usr/bin/env python3
def main():
	# Paramters to easily tune stuff
	params = {

		'limit'		: None, 		# None / 1231
		'verbose'	: True, 		# True / False
		'defuz' 	: "centroid",	# "centroid", "lom", "som"
		'trial'		: "max",		# "max", "rel", "high"

		'delimiter' : ';',

		'print_results': True,
		'results_path': "../res/results.txt",

		'datadump' 	: "../res/klachtendumpgemeente.csv",
		'testdump'  : "../res/testdump.csv",
		'traindump' : "../res/traindump.csv",
		'categories': "../res/categories/*.csv",
		'word_list' : "../res/categories/word_list/word_list.csv",

	}

	# Read test data
	dump = read_csv(params['testdump'],
					params['delimiter'])

	# Create rate object that creates
	# feature vectors for all emails
	rater =   Rater(params['categories'],
					params['word_list'])

	# Lists with categories used by the
	# rater object to rate the emails
	categories_list = rater.categories_list

	# Rows and rated generators to iterate
	# through rated and non-rated emails
	rows = ((row[0], tokenize(row[1])) for row in dump[1:])
	rated = ((row[0], row[1], rater.rate_email(row[1])) for row in rows)

	# Inputs for the Fuzzy Logic System
	inputs = [

		Input(feature[0], (0, 1), [
			TrapezoidalMF("low", -.2, -.1, 0, 0.5),
			TriangularMF("med", 0, 0.5, 1),
			TrapezoidalMF("high", 0.5, 1, 1.1, 1.2)
		]) for feature in categories_list

	]

	# Outputs for the Fuzzy Logic System
	outputs = [

		Output(feature[0], (0, 1), [
			TrapezoidalMF("low", -.2, -.1, 0, 0.5),
			TriangularMF("med", 0, 0.5, 1),
			TrapezoidalMF("high", 0.5, 1, 1.1, 1.2)
		]) for feature in categories_list

	]

	# Rules for the Fuzzy Logic System
	rules = [

		Rule(1, ["high", "", "", ""],
			"and", ["high", "", "", ""]),
		Rule(2, ["med", "", "", ""],
			"and", ["med", "", "", ""]),
		Rule(3, ["low", "", "", ""],
			"and", ["low", "", "", ""]),
		Rule(4, ["", "", "high", ""],
			"and", ["", "", "high", ""]),
		Rule(5, ["", "", "med", ""],
			"and", ["", "", "med", ""]),
		Rule(6, ["", "", "low", ""],
			"and", ["", "", "low", ""]),
		Rule(7, ["", "", "", "high"],
			"and", ["", "", "", "high"]),
		Rule(8, ["", "", "", "med"],
			"and", ["", "", "", "med"]),
		Rule(9, ["", "", "", "low"],
			"and", ["", "", "", "low"]),
		Rule(10, ["", "high", "", ""],
			"and", ["", "high", "", ""]),
		Rule(11, ["", "med", "", ""],
			"and", ["", "med", "", ""]),
		Rule(12, ["", "low", "", ""],
			"and", ["", "low", "", ""]),

		# Catches empties
		Rule(13, ["low", "low", "low", "low"],
			"and", ["high", "", "", ""]),

	]

	# Fuzzy Logic Classifier
	classifier = Classifier(
		inputs, outputs,
		rules, params
	)

	# Analyzes entire or parts of a classification
	# of the test dataset
	statistics = Statistics(params)
	statistics.start(rated, classifier)

# Cleans plain text into arrays of words
def tokenize(body):
	tokens = word_tokenize(body)
	tokens = [w.lower() for w in tokens]
	tokens = [w for w in tokens if len(w) > 2]
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	words = [word for word in stripped if word.isalpha()]
	stop_words = list(get_stop_words('nl'))
	nltk_words = list(stopwords.words('dutch'))
	stop_words.extend(nltk_words)
	words = [w for w in words if not w in stop_words]
	stemmer = SnowballStemmer("dutch")
	words = [stemmer.stem(word) for word in words]
	return words

# Reads comma separated file
def read_csv(filepath, delimiter=','):
	with open(filepath, 'r') as c:
		return [row for row in csv.reader(c, delimiter=delimiter,
			skipinitialspace=True)]

# Compares arrays of words and calculates a score
class Rater:
	def __init__(self, features, word_list):
		self.path = features
		self.word_list = read_csv(word_list)[0]
		self.len_word_list = len(self.word_list)
		self.categories_list = [
			(os.path.basename(fname).split('.')[0],
			read_csv(fname)[0])
			for fname in glob.glob(self.path)]
		self.categories_list.sort(key=lambda tup: tup[0])
	def corpus(self, email):
		words = [w for w in email if w in self.word_list]
		return np.c_[np.unique(words, return_counts=True)]
	def rate_words(self, email):
		c = self.corpus(email)
		c_len = len(c)
		for n, f in self.categories_list:
			f_len = len(f)
			c = np.c_[c, np.zeros(c_len)]
			for row in c:
				if (row[0] in f):
					# Alternative method purely looks at the document
					# row[-1:] = int(row[1]) / c_len

					# Main method takes category length vs word-list length
					# into account
					row[-1:] = int(row[1]) / c_len * (
						1 - (f_len / self.len_word_list))
		return c
	def rate_email(self, email):
		c = self.rate_words(email)
		ratings = dict()
		for i, category in enumerate(self.categories_list):
			agg = min(c[:,i + 2].astype(np.float).sum(), 1.0)
			ratings[category[0]] = float(format(agg, '.2f'))
		return ratings

# Classifies one or bulks of emails
class Statistics:
	def __init__(self, params):
		self.params = params
		self.iterations = 0
		self.success = 0
		self.template = "{label:19.19} | {c:19.19} | {success:7}"
		self.verbose = "score: {guess_score}, opposite: {opposite_score}, relative: {relative_score}"
	def print(self, classification, file):
		if self.params['verbose']:
			print(self.template.format(**classification), file=file)
			print(classification['r'], '\n', file=file)
			print(self.verbose.format(**classification), file=file)
		else:
			print(self.template.format(**classification), file=file)
			# print(classification['r'], file=file)
	def push(self, c):
		self.iterations += 1
		if c['correct_guess']:
			self.success += 1
	def start(self, rated, classifier):
		with open(self.params['results_path'], 'w', newline='') as f:
			if not self.params['print_results']:
				f = None
			print("%19s | %19s | %7s | %1s"
				% ("LABEL", "CLASS", "SUCCESS", "RATING"), file=f)
			for i, email in enumerate(rated):
				c = classifier.classify(email)
				self.push(c)
				self.print(c, file=f)
				if self.params['limit'] and i + 1 >= self.params['limit']:
					break
			print("\nTotal Success:", self.success, "/",
				self.iterations,
				"(" + str(round(self.success / self.iterations * 100, 1))
				+ "%)\n", file=f)

			if self.params['trial'] == "max":
				print("Trial 'max': (correctly guessed if class equals label)",
					file=f)
			elif self.params['trial'] == "rel":
				print("Trial 'rel': (correctly guessed if relative > 0.33, "
					+ "enable verbose for more output information)",
					file=f)
			elif self.params['trial'] == "high":
				print("Trial 'high': (correctly guessed if score > 0.75, "
					+ "enable verbose for more output information)",
					file=f)
		if self.params['print_results']:
			print("\nResults printed in file:", self.params['results_path'])

# Imports hidden at the bottom
import os
import csv
import glob
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')

from many_stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from __fuzzy_logic.classifier import *

# Calls main method
if __name__ =='__main__':
	main()
