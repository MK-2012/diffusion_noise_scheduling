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
    EFFECTIVE_BACTH_SIZE = 14
    REAL_BATCH_SIZE = 14
    gradient_accumulation_steps = int(ceil(EFFECTIVE_BACTH_SIZE / REAL_BATCH_SIZE))
    DEVICE = "cuda:0"
    NUM_EPOCHS = 14

    MovMNIST_frame_dataset = MovMNISTFrameDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_frame_dataloader = DataLoader(
        MovMNIST_frame_dataset,
        shuffle=True, batch_size=68
    )

    trainer_image = init_big_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=7,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_frame_dataloader),
        device=DEVICE,
        model_type="image",
        use_labels=True,
        noise_cov_matrix = torch.eye(20),
    )

    trainer_image.load_state(
        base_dir_path="./models/labeled_mov_mnist/frames_big/",
        suffix="last",
        load_optimizer=False, load_lr_sched=False, load_ema_model=True,
    )

    MovMNIST_dataset = MovMNISTDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_dataloader = DataLoader(MovMNIST_dataset, shuffle=True, batch_size=REAL_BATCH_SIZE)

    # sample_noise_corr = torch.load("./sample_corr_matrix.pt", map_location="cpu").float()
    trainer_video = init_big_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=NUM_EPOCHS,
        total_num_steps=100,
        beta_start=1.17e-3, # 1.17e-3
        beta_end=1.88e-1,   # 1.88e-1
        object_cnt = len(MovMNIST_dataloader),
        device=DEVICE,
        model_type="video",
        use_labels=True,
        cross_att_dim=4,
        noise_cov_matrix = progressive_noise(20),
    )

    trainer_video.load_weights_from(trainer_image.model_ref)
    trainer_video.load_weights_from(trainer_image.EMA_model, load_to="ema_model")
    # trainer_video.load_state("./models/trained/labeled_mov_mnist_progressive_noise_fixed", suffix="last", load_lr_sched=False)

    del MovMNIST_frame_dataset
    del MovMNIST_frame_dataloader
    del trainer_image
    # del model_frame
    # del optimizer_frame
    # del noise_scheduler_frame
    # del lr_scheduler_frame
    # del criterion_frame

    torch.cuda.empty_cache()
    collect()

    # trainer_video.compile(mode_ema=None)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    losses = trainer_video.fit(
        dataloader = MovMNIST_dataloader,
        save_path = "./models/labeled_mov_mnist/prog_noise_big/",
        num_epochs = NUM_EPOCHS,
        grad_accum_steps = gradient_accumulation_steps,
        class_free_guidance_threshhold = 0.0,
    )

    # losses = torch.cat([torch.load("./models/labeled_mov_mnist/prog_noise_big/losses.pt"), losses])
    torch.save(losses, "./models/labeled_mov_mnist/prog_noise_big/losses.pt")


if __name__ == "__main__":
    main()
