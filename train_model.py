import numpy as np
import scipy.stats as sps
from tqdm import tqdm
from torchinfo import summary # DEBUG
from functools import partial

from utils.utils import *
from utils.dataset_loaders import *
from models.basic_models import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.io import write_video, write_png
from diffusers import UNet3DConditionModel, DDPMScheduler, DDPMPipeline


if __name__ == "__main__":
    MovMNIST_dataset = MovMNISTDataset("./datasets/moving_mnist.npy")
    MovMNIST_dataloader = DataLoader(MovMNIST_dataset, shuffle=True, batch_size=2)

    model_video, noise_scheduler_video, optimizer_video, lr_scheduler_video, criterion_video = init_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=5,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_dataloader),
        device="cuda:1",
    )

    trainer = TrainableDiffusionModel(
        model_ref = model_video,
        optimizer_ref = optimizer_video,
        lr_scheduler_ref=lr_scheduler_video,
        noise_scheduler = noise_scheduler_video,
        criterion = criterion_video,
        device="cuda:1",
        model_type="video",
        cross_att_dim=4,
        noise_cov = lambda x: eye(x),
        EMA_start=7500,
    )

    test_losses = trainer.fit(
        dataloader = MovMNIST_dataloader,
        save_path = "./models/trained/mov_mnist_uncorr_noise_long/",
        num_epochs = 5,
    )

    torch.save(test_losses, "./models/trained/mov_mnist_uncorr_noise_long/losses.pt")
