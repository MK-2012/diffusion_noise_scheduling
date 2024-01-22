import numpy as np
from tqdm import tqdm
from torch import eye, ones, zeros, arange, diag_embed, diagonal, einsum, randn, randint, sqrt, pow, float32, no_grad, sin, tensor
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
	mult = alpha / np.sqrt(1 + alpha ** 2)
	mult_row = pow(mult, arange(1, length))
	ret = eye(length)
	for i in range(1, length):
		diagonal(ret, offset=i)[:] = mult_row[i - 1]
		diagonal(ret, offset=-i)[:] = mult_row[i - 1]
	return ret

class NormalVideoNoise:
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
			   self.a.repeat(list(shape[3:]) + [1]).permute(-1, *np.arange(len(shape) - 3))


# Training API

def train_simple_new(
	model,
	dataloader,
	noise_scheduler,
	optimizer,
	lr_scheduler,
	criterion,
	num_epochs,
	device="cpu",
	noise_cov=lambda x: eye(x),
	total_num_steps=1000,
):
	"""
	noise_cov -- matrix with the shape of video length or callable that receives video length and 
	             returns matrix
	"""

	losses = []
	for epoch in range(num_epochs):
		pbar = tqdm(dataloader)
		for i, (videos, labels) in enumerate(pbar):
			videos = videos[:, :, :25].to(device) # Attention!
			steps = randint(low=0, high=total_num_steps + 1, size=(videos.shape[0],), device=device)
			if callable(noise_cov):
				noise_gen = NormalVideoNoise(cov_matrix = noise_cov(videos.shape[2]))
			else:
				noise_gen = NormalVideoNoise(cov_matrix = noise_cov)
			noise = noise_gen.sample(videos.shape)
			noised_videos = noise_scheduler.add_noise(videos, noise, steps)
			hidden_states_encs = sin((labels.unsqueeze(-1) + 1) * arange(1, 24+1)).tile(1, videos.shape[2]).view(videos.shape[0], videos.shape[2], -1).to(device).float()
			predicted_noise = model(
				noised_videos,
				steps.to(device),
				hidden_states_encs,
			).sample
			loss = criterion(noise, predicted_noise)
			losses.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()

			pbar.set_postfix(MSE=loss.item())

			print("done once")

			break
		break

	return losses


def sample_videos(
	model,
	num_videos,
	video_length,
	noise_scheduler,
	prompts=None,
	pic_size=(240, 320),
	device="cpu",
	noise_cov=lambda x: eye(x),
):
	if callable(noise_cov):
		noise_gen = NormalVideoNoise(cov_matrix = noise_cov(video_length))
	else:
		noise_gen = NormalVideoNoise(cov_matrix = noise_cov)

	if prompts is None:
		prompts = ones(num_videos, video_length, 24, device=device, dtype=float32)

	with no_grad():
		sample = noise_gen.sample((num_videos, 3, video_length, pic_size[0], pic_size[1])).to(device)
		for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
			residual = model(sample, t, prompts).sample
			sample = noise_scheduler.step(residual, t, sample).prev_sample
	return sample
