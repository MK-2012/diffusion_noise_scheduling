import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils.training import *
from utils.dataset_loaders import *
from models.basic_models import *


def main():
    # gradient_accumulation_steps = 2
    # conf = ProjectConfiguration(automatic_checkpoint_naming=True)

    # accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, project_config=conf)

    MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist.npy")
    MovMNIST_frame_dataloader = DataLoader(MovMNIST_frame_dataset, shuffle=True, batch_size=32)

    model_frame, noise_scheduler_frame, optimizer_frame, lr_scheduler_frame, criterion_frame = init_mov_mnist_model_frame(
        lr_warmup_steps=100,
        num_epochs=1,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_frame_dataloader),
        device="cuda:1",
    )
    # model, optimizer, MovMNIST_frame_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, MovMNIST_frame_dataloader, lr_scheduler,
    # )

    trainer_image = TrainableDiffusionModel(
        model_ref = model_frame,
        optimizer_ref = optimizer_frame,
        lr_scheduler_ref=lr_scheduler_frame,
        noise_scheduler = noise_scheduler_frame,
        criterion = criterion_frame,
        device="cuda:1",
        model_type="image",
        cross_att_dim=4,
        EMA_start=2500,
    )

    trainer_image.load_state(base_dir_path="./models/trained/mov_mnist_frames_batch96/", suffix="8000",
                             load_optimizer=False, load_lr_sched=False, load_ema_model=False)



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

    trainer_video = TrainableDiffusionModel(
        model_ref = model_video,
        optimizer_ref = optimizer_video,
        lr_scheduler_ref=lr_scheduler_video,
        noise_scheduler = noise_scheduler_video,
        criterion = criterion_video,
        device="cuda:1",
        model_type="video",
        cross_att_dim=4,
        EMA_start=7500,
    )

    trainer_video.load_weights_from(trainer_image.model_ref)
    trainer_video.load_weights_from(trainer_image.model_ref, other_type="ema_model")

    del MovMNIST_frame_dataset
    del MovMNIST_frame_dataloader
    del trainer_image
    del model_frame
    del optimizer_frame
    del noise_scheduler_frame
    del lr_scheduler_frame
    del criterion_frame

    test_losses = trainer_video.fit(
        dataloader = MovMNIST_dataloader,
        save_path = "./models/trained/mov_mnist_uncorr_mnist_hot_start/",
        num_epochs = 5,
        grad_accum_steps = 16,
    )

    # accelerator.save(test_losses, "./models/trained/test/losses.pt")
    torch.save(test_losses, "./models/trained/mov_mnist_uncorr_mnist_hot_start/losses.pt")


if __name__ == "__main__":
    main()
