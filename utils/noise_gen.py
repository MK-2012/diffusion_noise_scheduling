from torch import diagonal, diag_embed, as_tensor, zeros, ones, arange, eye, einsum, pow, sqrt, randn
from torch.linalg import eigh

# Noises from the article
# Default alpha values taken from the article

def mixed_noise(length: int, alpha=1.0):
	ret = ones((length, length))
	al_2 = alpha ** 2
	ret *= al_2 / (1 + al_2)
	diagonal(ret)[:] = ones(length)
	return ret

def progressive_noise(length: int, alpha=2.0):
	mult = alpha / sqrt(1 + as_tensor(alpha ** 2))
	mult_row = pow(mult, arange(1, length))
	ret = eye(length)
	for i in range(1, length):
		diagonal(ret, offset=i)[:] = mult_row[i - 1]
		diagonal(ret, offset=-i)[:] = mult_row[i - 1]
	return ret

class NormalVideoNoise():
	"""Class for generating multivariate normal distribution"""

	__slots__ = "V", "a", "d"

	def __init__(self, mean = None, cov_matrix = None):
		assert mean is not None or cov_matrix is not None, "Some params must be not None"
		if cov_matrix is not None:
			assert cov_matrix.shape[-1] == cov_matrix.shape[-2], "Matrix must be square"
			self.d = cov_matrix.shape[-1]
			if mean is not None:
				assert cov_matrix.shape[-1] == mean.shape[-1], "Mean must have same size as cov matrix"
		else:
			self.d = mean.shape[-1]

		if cov_matrix is not None:
			L, Q = eigh(cov_matrix)
			self.V = Q @ diag_embed(sqrt(L))
		else:
			self.V = eye(self.d)
		if mean is not None:
			self.a = mean
		else:
			self.a = zeros(self.d)

	def sample(self, shape):
		return einsum("abj...,ij->abi...", randn(shape), self.V) + \
			   self.a.repeat(list(shape[3:]) + [1]).permute(-1, *arange(len(shape) - 3))