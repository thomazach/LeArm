import numpy as np

class matrix:
	def __init__ (self, lst):
		self.mat = np.array(lst)
		self.nrow, self.ncol = np.shape(self.mat)

	def norm(self):
		if self.ncol == 1 or self.nrow == 1:
			return np.linalg.norm(self.mat)
		else:
			raise ValueError("Tried to take the norm of a matrix. (need a vector)")
	
	def cross(self, other):
		if isinstance(self, matrix) & isinstance(other, matrix):
			return matrix(np.cross(self.mat, other.mat))

	def __add__ (self, other):
		if isinstance(self, matrix) & isinstance(other, matrix):
			return matrix(np.add(self.mat, other.mat))
	
	def __sub__(self, other):
		if isinstance(self, matrix) & isinstance(other, matrix):
			return matrix(self.mat - other.mat)
	
	def __radd__ (self, other):
		if isinstance(self, matrix) & isinstance(other, matrix):
			return matrix(np.add(self.mat, other.mat))

	def __mul__ (self, other):
		
		if isinstance(self, matrix) & isinstance(other, matrix):
			return matrix(np.matmul(self.mat, other.mat))
		elif isinstance(self, matrix) & (isinstance(other, int) or isinstance(other, float)):
			return matrix(other * self.mat)
		else:
			raise TypeError("Multiplication of non-matrix object with matrix object.")
	
	def __rmul__(self, other):
		if isinstance(self, matrix) & (isinstance(other, int) or isinstance(other, float)):
			return matrix(other * self.mat)
	
	def __str__(self):
		print("")
		return np.array_str(self.mat)
	
	def __repr__(self):
		return "\n" + np.array_str(self.mat)
	

# v1 = matrix([[6],[15],[4]])
# v2 =  matrix([[0], [0], [39.8]])

# print(v1-v2)