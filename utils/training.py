from tqdm import tqdm
from copy import deepcopy

from torch import eye, empty, arange, cat, as_tensor, randn, randint, uint8, float32, no_grad, sin, save, load
from torch.nn import MSELoss
from torch.cuda import empty_cache

from .noise_gen import *

class EMA():
	"""
	Exponential moving average class for training models
	"""

	__slots__ = "beta", "step", "activation_start"

	def __init__(self, beta, activation_start=3000):
		self.beta = beta
		self.step = 0
		self.activation_start = activation_start

	def update_average(self, old, new):
		if old is None:
			return new
		return self.beta * old + (1 - self.beta) * new

	def update_model_average(self, ma_model, current_model):
		for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = self.update_average(old_weight, up_weight)

	def step_ema(self, ema_model, model):
		if self.step < self.activation_start:
			self.reset_parameters(ema_model, model)
			self.step += 1
			return
		self.update_model_average(ema_model, model)
		self.step += 1

	def reset_parameters(self, ema_model, model):
		ema_model.load_state_dict(model.state_dict())


# Training API
class SimpleSaveCallbacker():
	__slots__ = "save_timer", "mandatory_save_period", "min_loss", "model_ref", "optimizer_ref", "lr_scheduler_ref", "ema_model_ref", "pbar", "save_path"

	def __init__(
		self,
		model_ref,
		optimizer_ref,
		save_path,
		lr_scheduler_ref=None,
		ema_model_ref=None,
		pbar=None,
		mandatory_save_period=500,
	):
		self.save_timer = 0
		self.mandatory_save_period = mandatory_save_period
		self.min_loss = np.inf
		self.model_ref = model_ref
		self.optimizer_ref = optimizer_ref
		self.lr_scheduler_ref = lr_scheduler_ref
		self.ema_model_ref = ema_model_ref
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
			if self.EMA_model is not None:
				save(self.EMA_model.state_dict(), self.save_path + "ema_model_best.pt")

		if self.pbar is not None:
			self.pbar.set_postfix(MSE=loss.item())

		if (self.save_timer % self.mandatory_save_period) == 0:
			save(self.model_ref.state_dict(), self.save_path + f"model_{self.save_timer}.pt")
			save(self.optimizer_ref.state_dict(), self.save_path + f"optimizer_{self.save_timer}.pt")
			if self.lr_scheduler_ref is not None:
				save(self.lr_scheduler_ref.state_dict(), self.save_path + f"scheduler_{self.save_timer}.pt")
			if self.EMA_model is not None:
				save(self.EMA_model.state_dict(), self.save_path + f"ema_model_{self.save_timer}.pt")


class TrainableDiffusionModel():
	"""
	Class for training most of my diffusion models
	"""

	__slots__ = "model_ref", "optimizer_ref", "noise_scheduler", "criterion", "device", "model_type", "cross_att_dim",\
				"noise_cov", "lr_scheduler_ref", "EMA_model", "EMA_sched"

	available_model_types = {"video", "image"}

	def __init__(
		self,
		model_ref,
		optimizer_ref,
		noise_scheduler,
		criterion = MSELoss(),
		device="cpu",
		model_type="video",
		cross_att_dim=24,
		noise_cov = lambda x: eye(x),
		lr_scheduler_ref=None,
		EMA_coeff=0.995,
		EMA_start=2500,
	):

		# basic asserts
		assert model_type in self.available_model_types, "Model type should be 'image' or 'video'."

		# basic fields init
		self.model_ref = model_ref
		self.optimizer_ref = optimizer_ref
		self.noise_scheduler = noise_scheduler
		self.criterion = criterion
		self.device = device
		self.model_type = model_type
		self.cross_att_dim = cross_att_dim
		self.noise_cov = noise_cov
		self.lr_scheduler_ref = lr_scheduler_ref

		# details of diferrent model types
		# if self.model_type == "image":
		# 	self.step_func = self._image_train_step
		# elif self.model_type == "video":
		# 	self.step_func = self._video_train_step

		# EMA setup
		if EMA_start <= 0 or EMA_coeff <= 0.0 or EMA_coeff >= 1.0:
			self.EMA_model = None
			self.EMA_sched = None
		else:
			self.EMA_model = deepcopy(self.model_ref).eval().requires_grad_(False)
			self.EMA_sched = EMA(beta=EMA_coeff, activation_start=EMA_start)

	# def load_image_layers(self, image_model, noise_scale=1e-2):


	def _sample_noise(self, shape):
		if self.model_type == "video":
			if callable(self.noise_cov):
				noise_gen = NormalVideoNoise(cov_matrix = self.noise_cov(shape[2]))
			else:
				noise_gen = NormalVideoNoise(cov_matrix = self.noise_cov)
			return noise_gen.sample(shape).to(self.device)

		elif self.model_type == "image":
			return randn(shape, device=self.device, dtype=float32)

	def _sin_embeddings(self, shape):
		if self.model_type == "video":
			return sin(arange(1, shape[2] + 1, device=self.device).view(-1, 1) * arange(1, self.cross_att_dim + 1, device=self.device)).tile(shape[0]).view(shape[0], shape[2], -1)
		elif self.model_type == "image":
			return sin(arange(1, self.cross_att_dim + 1, device=self.device)).tile(shape[0]).view(shape[0], 1, -1)

	def _train_step(self, batch):
		"""
		noise_cov -- matrix with the shape of video length or callable that receives video length and
					returns matrix
		"""

		steps = randint(low=0, high=len(self.noise_scheduler.timesteps), size=(batch.shape[0],), device=self.device)
		noise = self._sample_noise(batch.shape)
		noised_videos = self.noise_scheduler.add_noise(batch, noise, steps)
		hidden_states_encs = self._sin_embeddings(batch.shape)
		return noise, self.model_ref(
					noised_videos,
					steps,
					hidden_states_encs,
				).sample

	def _one_epoch(self, dataloader, end_processor):
		pbar = tqdm(dataloader)
		end_processor.set_pbar(pbar)
		losses = []
		for i, (batch, labels) in enumerate(pbar):
			batch = batch.to(self.device)
			noise, predicted_noise = self._train_step(batch)

			loss = self.criterion(noise, predicted_noise)
			end_processor(loss)
			if self.EMA_model is not None:
				self.EMA_sched.step_ema(self.EMA_model, self.model_ref)

			losses.append(loss.item())
		return losses

	def fit(self, dataloader, save_path, num_epochs=2, end_processor=SimpleSaveCallbacker):
		# initial fitting setup
		losses = empty(0, len(dataloader))
		end_proc = end_processor(model_ref = self.model_ref, optimizer_ref = self.optimizer_ref, save_path=save_path,
								 lr_scheduler_ref=self.lr_scheduler_ref, ema_model=self.EMA_model)

		# Going through epochs
		for _ in range(num_epochs):
			new_losses = self._one_epoch(dataloader, end_proc)
			losses = cat([losses, as_tensor(new_losses).unsqueeze(0)], dim=0)

		# saving last versions of models
		save(self.model_ref.state_dict(), self.save_path + "model_last.pt")
		save(self.optimizer_ref.state_dict(), self.save_path + "optimizer_last.pt")
		if self.lr_scheduler_ref is not None:
			save(self.lr_scheduler_ref.state_dict(), self.save_path + "scheduler_last.pt")
		if self.EMA_model is not None:
			save(self.EMA_model.state_dict(), self.save_path + "ema_model_last.pt")

		return losses

	def load_state(self, base_dir_path, suffix="best", load_moel=True, load_optimizer=True, load_lr_sched=True):
		if load_moel:
			self.model_ref.load_state_dict(load(base_dir_path + "model_" + suffix + ".pt"))
		if load_optimizer:
			self.optimizer_ref.load_state_dict(load(base_dir_path + "optimizer_" + suffix + ".pt"))
		if load_lr_sched:
			self.lr_scheduler_ref.load_state_dict(load(base_dir_path + "scheduler_" + suffix + ".pt"))
	
	def sample(self, num_samples, prompts=None, pic_size=(64, 64), num_channels=1, video_length=None):
		shape = [num_samples, num_channels, *pic_size]
		if self.model_type == "video":
			shape.insert(2, video_length)

		with no_grad():
			sample = self._sample_noise(shape)
			if prompts is None:
				prompts = self._sin_embeddings(shape)

			if self.EMA_model is not None:
				for t in tqdm(self.noise_scheduler.timesteps):
					residual = self.EMA_model(sample, t, prompts).sample
					sample = self.noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample
			else:
				for t in tqdm(self.noise_scheduler.timesteps):
					residual = self.model_ref(sample, t, prompts).sample
					sample = self.noise_scheduler.step(model_output=residual, timestep=t, sample=sample).prev_sample

		return ((sample.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).to(uint8)
