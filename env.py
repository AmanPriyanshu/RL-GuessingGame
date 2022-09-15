import numpy as np

class Environment:
	def __init__(self, num_chits, majority_chits=None):
		self.num_chits = num_chits
		if majority_chits is None:
			majority_chits = self.num_chits//2+1
		self.majority_chits = majority_chits

	def pick_up(self):
		return 1 if np.random.random()<=self.majority_chits/self.num_chits else 0

	def check_validation(self, choice):
		return 1 if choice=="A" else 0