import numpy as np
from tqdm import tqdm
from gc import collect
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch import eye, ones, zeros, arange, diag_embed, diagonal, einsum, randn, randn_like, randint, sqrt, pow, float32, no_grad, sin, save, uint8
from torch.cuda import empty_cache
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
class SimpleSaveCallbacker():
	__slots__ = "save_timer", "mandatory_save_period", "min_loss", "model_ref", "optimizer_ref", "lr_scheduler_ref", "pbar", "save_path"

	def __init__(self, model_ref, optimizer_ref, save_path, lr_scheduler_ref=None, pbar=None, mandatory_save_period=500):
		self.save_timer = 0
		self.mandatory_save_period = mandatory_save_period
		self.min_loss = np.inf
		self.model_ref = model_ref
		self.optimizer_ref = optimizer_ref
		self.lr_scheduler_ref = lr_scheduler_ref
		self.pbar = pbar
		self.save_path = save_path

	def set_pbar(self, new_pbar):
		self.pbar = new_pbar

	def __call__(self, loss):
		self.save_timer += 1

		self.optimizer_ref.zero_grad()
		loss.backward()
		self.optimizer_ref.step()
		self.lr_scheduler_ref.step()

		if loss.item() <= self.min_loss:
			self.min_loss = loss.item()
			save(self.model_ref.state_dict(), self.save_path + "model_best.pt")
			save(self.optimizer_ref.state_dict(), self.save_path + "optimizer_best.pt")
			if self.lr_scheduler_ref is not None:
				save(self.lr_scheduler_ref.state_dict(), self.save_path + "scheduler_best.pt")

		if self.pbar is not None:
			self.pbar.set_postfix(MSE=loss.item())

		if (self.save_timer % self.mandatory_save_period) == 0:
			save(self.model_ref.state_dict(), self.save_path + f"model_{self.save_timer}.pt")
			save(self.optimizer_ref.state_dict(), self.save_path + f"optimizer_{self.save_timer}.pt")
			if self.lr_scheduler_ref is not None:
				save(self.lr_scheduler_ref.state_dict(), self.save_path + f"scheduler_{self.save_timer}.pt")
		
		# if (save_timer % 50) == 0:
		# 	empty_cache()
		# 	collect()


def train_simple_new(
	model,
	dataloader,
	noise_scheduler,
	criterion,
	num_epochs,
	device="cpu",
	noise_cov=lambda x: eye(x),
	cross_att_dim=24,
	end_processor=None,
):
	"""
	noise_cov -- matrix with the shape of video length or callable that receives video length and
	             returns matrix
	end_processor -- instance of an object that must implement __call__(self, loss). It is called at the
					 end of each training step
	"""

	losses = []
	for epoch in range(num_epochs):
		pbar = tqdm(dataloader)
		end_processor.set_pbar(pbar)
		for i, (videos, labels) in enumerate(pbar):
			# videos = videos[:, :, :25].to(device) # Attention!
			videos = videos.to(device)
			steps = randint(low=0, high=len(noise_scheduler.timesteps), size=(videos.shape[0],), device=device)
			if callable(noise_cov):
				noise_gen = NormalVideoNoise(cov_matrix = noise_cov(videos.shape[2]))
			else:
				noise_gen = NormalVideoNoise(cov_matrix = noise_cov)
			noise = noise_gen.sample(videos.shape).to(device)
			noised_videos = noise_scheduler.add_noise(videos, noise, steps)
			# hidden_states_encs = sin((labels.unsqueeze(-1) + 1) * arange(1, cross_att_dim+1)).tile(1, videos.shape[2]).view(videos.shape[0], videos.shape[2], -1).to(device).float()
			hidden_states_encs = sin(arange(1, videos.shape[2] + 1, device=device).view(-1, 1) * arange(1, cross_att_dim + 1, device=device)).tile(videos.shape[0]).view(videos.shape[0], videos.shape[2], -1)
			predicted_noise = model(
				noised_videos,
				steps,
				hidden_states_encs,
			).sample

			loss = criterion(noise, predicted_noise)
			end_processor(loss)

			losses.append(loss.item())
	return losses


def train_images(
	model,
	dataloader,
	noise_scheduler,
	criterion,
	num_epochs,
	device="cpu",
	cross_att_dim=24,
	end_processor=None,
):
	"""
	noise_cov -- matrix with the shape of video length or callable that receives video length and
	             returns matrix
	end_processor -- instance of an object that must implement __call__(self, loss). It is called at the
					 end of each training step
	"""

	losses = []
	for epoch in range(num_epochs):
		pbar = tqdm(dataloader)
		end_processor.set_pbar(pbar)
		for i, (images, labels) in enumerate(pbar):
			images = images.to(device)
			steps = randint(low=0, high=len(noise_scheduler.timesteps), size=(images.shape[0],), device=device)
			noise = randn_like(images, device=device, dtype=float32)
			noised_images = noise_scheduler.add_noise(images, noise, steps)
			hidden_states_encs = sin(arange(1, cross_att_dim + 1, device=device)).tile(images.shape[0]).view(images.shape[0], 1, -1)
			predicted_noise = model(
				noised_images,
				steps,
				hidden_states_encs,
			).sample

			loss = criterion(noise, predicted_noise)
			end_processor(loss)

			losses.append(loss.item())
	return losses


# Simple callback for drawing frames from the first video of the batch
def draw_single_vid_frames(videos):
	clear_output(wait=True)

	video = ((videos[0].detach().cpu().clamp(-2, 2) / 2 + 1) * 255 / 2).detach().cpu().to(uint8).permute(1, 2, 3, 0)

	step = video.shape[0] // 5

	fig, ax = plt.subplots(nrows=1, ncols=5, constrained_layout=True)
	fig.set_size_inches(12, 4)

	for i in range(5):
		ax[i].imshow(video[i * step], cmap="grey")
		ax[i].axis("off")

	plt.show()


def sample_videos(
	model,
	num_videos,
	video_length,
	noise_scheduler,
	prompts=None,
	pic_size=(240, 320),
	sample=None,
	device="cpu",
	noise_cov=lambda x: eye(x),
	cross_att_dim=24,
	channel_num=3,
	display_callback=None,
):
	if callable(noise_cov):
		noise_gen = NormalVideoNoise(cov_matrix = noise_cov(video_length))
	else:
		noise_gen = NormalVideoNoise(cov_matrix = noise_cov)

	if prompts is None:
		prompts = sin(arange(1, video_length + 1, device=device).view(-1, 1) * arange(1, cross_att_dim + 1, device=device)).tile(num_videos).view(num_videos, video_length, -1)

	with no_grad():
		if sample is None:
			sample = noise_gen.sample((num_videos, channel_num, video_length, pic_size[0], pic_size[1])).to(device)
		if display_callback is None:
			for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
				residual = model(sample, t, prompts).sample
				sample = noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample
		else:
			for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
				residual = model(sample, t, prompts).sample
				sample = noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample

				display_callback(sample)

	return sample


def sample_images(
	model,
	num_images,
	noise_scheduler,
	prompts=None,
	pic_size=(64, 64),
	sample=None,
	device="cpu",
	cross_att_dim=24,
	channel_num=1,
	display_callback=None,
):
	if prompts is None:
		prompts = sin(arange(1, cross_att_dim + 1, device=device)).tile(num_images).view(num_images, 1, -1)

	with no_grad():
		if sample is None:
			sample = randn(num_images, channel_num, pic_size[0], pic_size[1], device=device, dtype=float32)
		if display_callback is None:
			for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
				residual = model(sample, t, prompts).sample
				sample = noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample
		else:
			for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
				residual = model(sample, t, prompts).sample
				sample = noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample

				display_callback(sample)

	return sample
