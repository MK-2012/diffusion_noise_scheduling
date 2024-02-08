import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils.training import *
from utils.dataset_loaders import *
from models.basic_models import *


def main():
    gradient_accumulation_steps = 2
    conf = ProjectConfiguration(automatic_checkpoint_naming=True)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, project_config=conf)

    MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist.npy")
    MovMNIST_frame_dataloader = DataLoader(MovMNIST_frame_dataset, shuffle=True, batch_size=32)

    model, noise_scheduler, optimizer, lr_scheduler, criterion = init_mov_mnist_model_frame(
        lr_warmup_steps=100,
        num_epochs=1,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_frame_dataloader) // 2,
        device="cpu",
    )
    model, optimizer, MovMNIST_frame_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, MovMNIST_frame_dataloader, lr_scheduler,
    )

    trainer = TrainableDiffusionModel(
        accelerator_ref = accelerator,
        model_ref = model,
        optimizer_ref = optimizer,
        lr_scheduler_ref=lr_scheduler,
        noise_scheduler = noise_scheduler,
        criterion = criterion,
        model_type="image",
        cross_att_dim=4,
        EMA_start=5000,
    )

    test_losses = trainer.fit(
        dataloader = MovMNIST_frame_dataloader,
        save_path = "./models/trained/test/",
        num_epochs = 1,
    )

    accelerator.save(test_losses, "./models/trained/test/losses.pt")


if __name__ == "__main__":
    main()
