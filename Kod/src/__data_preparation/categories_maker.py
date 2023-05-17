#!/usr/bin/env python3
import os
import csv
import math
from collections import Counter
from __data_preparation.utils import *

class Tfidf:
	def __init__(self):
		self.n_containing_dict = dict()
	def tf(self, word, row):
		return row.count(word) / len(row)
	def n_containing(self, word, rows):
		if (word in self.n_containing_dict):
			return self.n_containing_dict[word]
		n = sum(1 for row in rows if word in row)
		self.n_containing_dict[word] = n
		return n
	def idf(self, word, rows):
		return math.log(len(rows) / (1 + self.n_containing(word, rows)))
	def tfidf(self, word, row, rows):
		return self.tf(word, row) * self.idf(word, rows)

class Corpus:
	"""Anlamlı kelimeleri filtreler ve labella birlikte tutar."""
	def __init__(self, params):
		self.rows = None
		self.categories = None
		self.process(params)

	# Starts steps of creating category lists
	def process(self, params):
		self.read_dump(params)
		self.count_distinct_categories()
		self.tokenize()
		self.filter_categories(params)

	# Reads the train datadump
	def read_dump(self, params):
		with open(params['traindump'], 'r') as c:
			reader = csv.reader(c,
				delimiter=params['delimiter'],
				skipinitialspace=True)
			self.rows = [row for row in reader][1:]

	# Counts distinct categories
	def count_distinct_categories(self):
		self.categories = list(set([row[0] for row in self.rows]))

	# Tokenizes and cleans email bodies
	def tokenize(self):
		for row in self.rows:
			row[1] = tokenize(row[1])

	# Creates lists of words, per category, with tf/idf score above threshold
	def filter_categories(self, params):
		if not os.path.exists(params['categories_path']):
			os.makedirs(params['categories_path'])
		if not os.path.exists(params['word_list_path']):
			os.makedirs(params['word_list_path'])
		word_list = []
		common_word_list = []

		# After folders are created, start tf/idf
		print("Starting tf/idf process, this may take a while...")
		tfidf = Tfidf()
		for category in self.categories:
			print("Category:", category, "- threshold:", params['threshold'])
			rows = [row for row in self.rows if category == row[0]]
			favorite_words = set(self.tfidf(rows, tfidf, params))
			print(category + ":", len(favorite_words))
			word_list += favorite_words
			common_word_list = intersection(common_word_list, favorite_words)
			generate_csv_from_array(params['categories_path'] + category.lower() + ".csv", favorite_words)

		# Creates final word_list, a union set of all category lists
		generate_csv_from_array(
			params['word_list_path'] + "word_list.csv",
			set([x for x in word_list if x not in common_word_list]))

	# Extracts words with tf/idf score above threshold
	def tfidf(self, rows, tfidf, params):
		favorite_words = []
		for i, row in enumerate(rows):
			scores = {word: tfidf.tfidf(word, row[1], [r[1] for r in self.rows]) for word in row[1]}
			best_words = sorted(scores.items(), key=lambda x: x[1],
				reverse=True)
			for word, score in best_words:
				if (score >= params['threshold']):
					if (params['verbose']):
						print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
					favorite_words.append(word)
		return favorite_words
