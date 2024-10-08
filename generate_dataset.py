from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

from utils.noise_gen import *
from utils.training import *
from utils.dataset_loaders import *
from .basic_models import *

from torch import eye, cat, save, empty
from torch.utils.data import DataLoader


BATCH_SIZE = 10

def generate_dataset(base_dataloader, model, device, noise_cov_type="uncorr", save_path=""):
    match noise_cov_type:
        case "uncorr":
            noise_cov = lambda x: eye(x)
        case "weak_mixed":
            noise_cov = partial(mixed_noise, alpha=0.1)
        case "mixed":
            noise_cov = mixed_noise
        case "progressive":
            noise_cov = progressive_noise

    i = 0
    dataset = empty(0, 1, 20, 64, 64, device="cpu")
    # dataset = load(save_path, map_location="cpu")
    for _, labels in tqdm(base_dataloader):
        i += 1
        labels = labels.to(device)
        objects = model.sample(
            num_samples=BATCH_SIZE,
            video_length=20,
            prompts=labels,
            override_noise_cov = noise_cov,
            disable_tqdm=True,
            convert_uint=False,
        ).detach().cpu()
        dataset = cat([dataset, objects], dim=0)
        # print(f"Epoch num {i + 1}")

        save(dataset, save_path)


if __name__ == "__main__":
    parser = ArgumentParser(    
        prog='Dataset generator',
    )
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--noise_type", choices=["uncorr", "mixed", "progressive"], required=True)
    parser.add_argument("--not_compile", action="store_true")

    MovMNIST_dataset = MovMNISTDataset("./datasets/moving_mnist_labeled/")
    MovMNIST_dataloader = DataLoader(MovMNIST_dataset, shuffle=False, batch_size=BATCH_SIZE)

    args = parser.parse_args()
    dev = args.device

    model_video, noise_scheduler_video, optimizer_video, lr_scheduler_video, criterion_video = init_big_mov_mnist_model(
        lr_warmup_steps=100,
        num_epochs=1,
        beta_start=1.17e-3,
        beta_end=1.88e-1,
        object_cnt = len(MovMNIST_dataloader),
        device=dev,
        model_type="video",
        use_labels=True,
    )
    
    sampler = TrainableDiffusionModel(
        model_ref = model_video,
        optimizer_ref = optimizer_video,
        lr_scheduler_ref=lr_scheduler_video,
        noise_scheduler = noise_scheduler_video,
        criterion = criterion_video,
        device=dev,
        model_type="video",
        EMA_start=5000,
    )

    print(f"Loading weights: {args.weights_path}")
    sampler.load_state(base_dir_path=args.weights_path, suffix="last", load_optimizer=False, load_lr_sched=False, load_ema_model=True)

    if not args.not_compile:
        sampler.compile(mode_model=None, mode_ema="max-autotune")

    weights_noise_type = args.weights_path.split("/")[-1][18:]
    dataset = generate_dataset(
        base_dataloader=MovMNIST_dataloader,
        model=sampler,
        device=dev,
        save_path=f"./results/MovMNIST/big_sets/{weights_noise_type}/{args.noise_type}_inference_noise.pt",
    )

    # weights_noise_type = args.weights_path.split("/")[-1][18:]
    # save(dataset, f"./results/MovMNIST/big_sets/{weights_noise_type}/{args.noise_type}_inference_noise.pt")
