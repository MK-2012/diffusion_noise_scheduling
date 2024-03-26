import torch
from numpy import ceil
from gc import collect
from torch.utils.data import DataLoader

from utils.training import *
from utils.dataset_loaders import *
from models.basic_models import *

from utils.noise_gen import *

from functools import partial


def main():
    EFFECTIVE_BACTH_SIZE = 256
    REAL_BATCH_SIZE = 256
    gradient_accumulation_steps = int(ceil(EFFECTIVE_BACTH_SIZE / REAL_BATCH_SIZE))
    DEVICE = "cuda:5"

    MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_frame_dataloader = DataLoader(
        MovMNIST_frame_dataset,
        shuffle=True, batch_size=REAL_BATCH_SIZE
    )

    model_frame, noise_scheduler_frame, optimizer_frame, lr_scheduler_frame, criterion_frame = init_big_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=8,
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
        EMA_start=750,
    )

    # test_losses = trainer_image.fit(
    #     dataloader = MovMNIST_frame_dataloader,
    #     save_path = "./models/trained/labeled_mov_mnist_frames/",
    #     num_epochs = 7,
    #     grad_accum_steps = 1,
    # )

    # torch.save(test_losses, "./models/trained/labeled_mov_mnist_frames/losses.pt")

    # trainer_image.load_state(
    #     base_dir_path="./models/trained/labeled_mov_mnist_frames/",
    #     suffix="last",
    #     load_optimizer=False, load_lr_sched=False, load_ema_model=True,
    # )

    test_losses = trainer_image.fit(
        dataloader = MovMNIST_frame_dataloader,
        save_path = "./models/trained/labeled_mov_mnist_frames_big/",
        num_epochs = 8,
        grad_accum_steps = gradient_accumulation_steps,
        class_free_guidance_threshhold = 3e-2,
    )

    torch.save(test_losses, "./models/trained/labeled_mov_mnist_frames_big/losses.pt")


if __name__ == "__main__":
    main()
