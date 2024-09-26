import torch
from numpy import ceil
from gc import collect
from torch.utils.data import DataLoader

from utils.training import *
from utils.dataset_loaders import *
from basic_models import *

from utils.noise_gen import *

from functools import partial


def main():
    EFFECTIVE_BACTH_SIZE = 512
    REAL_BATCH_SIZE = 512
    gradient_accumulation_steps = int(ceil(EFFECTIVE_BACTH_SIZE / REAL_BATCH_SIZE))
    DEVICE = "cuda:1"
    NUM_EPOCHS = 14

    MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_frame_dataloader = DataLoader(
        MovMNIST_frame_dataset,
        shuffle=True, batch_size=REAL_BATCH_SIZE
    )

    trainer_image = init_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=NUM_EPOCHS,
        total_num_steps=100,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_frame_dataloader),
        device=DEVICE,
        model_type="image",
        use_labels=True,
        noise_cov_matrix = torch.eye(20),
        EMA_start=400,
    )

    losses = trainer_image.fit(
        dataloader = MovMNIST_frame_dataloader,
        save_path = "./models/labeled_mov_mnist/frames/basic",
        num_epochs = NUM_EPOCHS,
        grad_accum_steps = gradient_accumulation_steps,
        class_free_guidance_threshhold = 0.0,
    )

    torch.save(losses, "./models/labeled_mov_mnist/frames/basic/losses.pt")


if __name__ == "__main__":
    main()
