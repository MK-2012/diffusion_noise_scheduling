import torch
from numpy import ceil
from gc import collect
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils.training import *
from utils.dataset_loaders import *
from basic_models import *

from utils.noise_gen import *


def main():
	GPU_NUM = 1
	EFFECTIVE_BACTH_SIZE = 64
	REAL_BATCH_SIZE = 8
	gradient_accumulation_steps = int(ceil(EFFECTIVE_BACTH_SIZE / REAL_BATCH_SIZE))
	conf = ProjectConfiguration(automatic_checkpoint_naming=True)

	accelerator = Accelerator(
		gradient_accumulation_steps=gradient_accumulation_steps,
		project_config=conf,
	)
	DEVICE = accelerator.device

	MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist_labeled/")
	MovMNIST_frame_dataloader = DataLoader(
		MovMNIST_frame_dataset,
		shuffle=True, batch_size=68
	)

	model_frame, noise_scheduler_frame, optimizer_frame, lr_scheduler_frame, criterion_frame = init_big_mov_mnist_model(
		lr_warmup_steps=100,
		num_epochs=7,
		beta_start=1.17e-3,
		beta_end=1.88e-1,
		object_cnt = len(MovMNIST_frame_dataloader),
		device=DEVICE,
		model_type="image",
		use_labels=True,
	)

	trainer_image = TrainableDiffusionModel(
		model_ref = model_frame,
		optimizer_ref = optimizer_frame,
		lr_scheduler_ref=lr_scheduler_frame,
		noise_scheduler = noise_scheduler_frame,
		criterion = criterion_frame,
		device=DEVICE,
		model_type="image",
		EMA_start=2500,
	)

	# test_losses = trainer_image.fit(
	#     dataloader = MovMNIST_frame_dataloader,
	#     save_path = "./models/trained/labeled_mov_mnist_frames/",
	#     num_epochs = 7,
	#     grad_accum_steps = 1,
	# )

	# torch.save(test_losses, "./models/trained/labeled_mov_mnist_frames/losses.pt")

	trainer_image.load_state(
		base_dir_path="./models/trained/labeled_mov_mnist_frames_big/",
		suffix="last",
		load_optimizer=False, load_lr_sched=False, load_ema_model=True,
	)

	MovMNIST_dataset = MovMNISTDataset("./datasets/moving_mnist_labeled/")
	MovMNIST_dataloader = DataLoader(MovMNIST_dataset, shuffle=True, batch_size=REAL_BATCH_SIZE)

	model, noise_scheduler, optimizer, lr_scheduler, criterion = init_big_mov_mnist_model(
		lr_warmup_steps=100,
		num_epochs=7,
		beta_start=1.17e-3,
		beta_end=1.88e-1,
		object_cnt = len(MovMNIST_dataloader),
		device=DEVICE,
		model_type="video",
		use_labels=True,
		cross_att_dim=4,
	)

	model, optimizer, MovMNIST_dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, MovMNIST_dataloader, lr_scheduler,
	)

	trainer_video = TrainableDiffusionModel(
		accelerator_ref=accelerator,
		model_ref = model,
		optimizer_ref = optimizer,
		lr_scheduler_ref=lr_scheduler,
		noise_scheduler = noise_scheduler,
		criterion = criterion,
		device=DEVICE,
		model_type="video",
		EMA_start=1200,
		noise_cov=lambda x: torch.eye(x),
	)

	trainer_video.load_weights_from(trainer_image.model_ref)
	trainer_video.load_weights_from(trainer_image.EMA_model, load_to="ema_model")

	del MovMNIST_frame_dataset
	del MovMNIST_frame_dataloader
	del trainer_image
	del model_frame
	del optimizer_frame
	del noise_scheduler_frame
	del lr_scheduler_frame
	del criterion_frame

	torch.cuda.empty_cache()
	collect()

	test_losses = trainer_video.fit(
		dataloader = MovMNIST_dataloader,
		save_path = "./models/trained/labeled_mov_mnist_uncorr_noise_big/",
		num_epochs = 7,
		class_free_guidance_threshhold = 3e-2,
	)

	accelerator.save(test_losses, "./models/trained/labeled_mov_mnist_uncorr_noise_big/losses.pt")
	# torch.save(test_losses, "./models/trained/labeled_mov_mnist/losses.pt")


if __name__ == "__main__":
	main()
