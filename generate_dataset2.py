from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

from utils.noise_gen import *
from utils.training import *
from utils.dataset_loaders import *
from basic_models import *
from torchvision.io import write_video

import torch
from torch import eye, cat, save, empty
from torch.utils.data import DataLoader


def generate_samples(labels, cov_function, obj_cnt: int, weights_name: str, device: str="cuda:0", n=1000, batch_size=10):
    sampler = init_big_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=14,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = obj_cnt,
        device=device,
        model_type="video",
        use_labels=True,
        cross_att_dim=4,
        noise_cov_matrix = cov_function(20),
    )

    sampler.load_state(base_dir_path=f"./models/labeled_mov_mnist/{weights_name}/", suffix="last",
                               load_optimizer=False, load_lr_sched=False, load_ema_model=True)
    sampler.EMA_model.cross_att_dim = 4
    sampler.EMA_model.need_time_embs = False

    sampler.compile(mode_model=None, mode_ema="default")
    for i in tqdm(range(n // batch_size)):
        objects = sampler.sample(
            num_samples=batch_size,
            video_length=20,
            prompts = labels[i * batch_size:(i + 1) * batch_size],
            override_noise_cov = lambda x: cov_function(x),
            disable_tqdm=True,
            num_inference_steps=100,
        )

        for j, v in enumerate(objects):
            write_video(
                f"./generation_comparison/{weights_name}/_{i * BATCH_SIZE + j}.mp4",
                v.repeat(3, 1, 1, 1).permute(1, 2, 3, 0),
                fps=7,
            )


if __name__ == "__main__":
    MovMNIST_dataset = MovMNISTDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_dataloader = DataLoader(MovMNIST_dataset, shuffle=True, batch_size=14)

    N = 1000
    BATCH_SIZE=10
    labels = torch.randint(low=0, high=55, size=(N,))

    dev = "cuda:0"


    # Uncorr noise
    generate_samples(labels=labels, obj_cnt=len(MovMNIST_dataset), weights_name="uncorr_noise_big", device=dev, n=N, batch_size=BATCH_SIZE, cov_function=torch.eye)


    # Progressive noise
    generate_samples(labels=labels, obj_cnt=len(MovMNIST_dataset), weights_name="prog_noise_big", device=dev, n=N, batch_size=BATCH_SIZE, cov_function=progressive_noise)
