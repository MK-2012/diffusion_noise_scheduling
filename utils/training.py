from numpy import inf
from tqdm import tqdm
from copy import deepcopy
from os.path import join
from inspect import Signature

from torch import (
    eye,
    empty,
    cat,
    as_tensor,
    rand,
    randn,
    randint,
    uint8,
    float32,
    no_grad,
    save,
    load,
    set_float32_matmul_precision,
)
from torch import max as t_max
from torch import abs as t_abs
from torch.linalg import inv as m_inv
from torch import compile as comp
from torch.nn import MSELoss
from torch.cuda import empty_cache

from .noise_gen import *


class EMA:
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
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
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
class SimpleSaveCallbacker:
    __slots__ = (
        "save_timer",
        "mandatory_save_period",
        "min_loss",
        "model_ref",
        "optimizer_ref",
        "accelerator_ref",
        "lr_scheduler_ref",
        "ema_model_ref",
        "pbar",
        "save_path",
        "grad_accum_steps",
    )

    def __init__(
        self,
        model_ref,
        optimizer_ref,
        save_path,
        accelerator_ref=None,
        lr_scheduler_ref=None,
        ema_model_ref=None,
        pbar=None,
        mandatory_save_period=500,
        grad_accum_steps=1,
    ):
        self.save_timer = 0
        self.mandatory_save_period = mandatory_save_period
        self.min_loss = inf
        self.model_ref = model_ref
        self.optimizer_ref = optimizer_ref
        self.accelerator_ref = accelerator_ref
        self.lr_scheduler_ref = lr_scheduler_ref
        self.ema_model_ref = ema_model_ref
        self.pbar = pbar
        self.save_path = save_path
        self.grad_accum_steps = grad_accum_steps

    def set_pbar(self, new_pbar):
        self.pbar = new_pbar

    def _unwrap(self, obj):
        if self.accelerator_ref is None:
            return obj
        else:
            return self.accelerator_ref.unwrap_model(obj)

    def _save(self, obj, path):
        if self.accelerator_ref is None:
            save(self._unwrap(obj.state_dict()), path)
        else:
            self.accelerator_ref.save(self._unwrap(obj), path)

    def _save_everything(self, suffix="best"):
        if self.accelerator_ref is not None:
            self.accelerator_ref.wait_for_everyone()
        self._save(self.model_ref, join(self.save_path, f"model_{suffix}.pt"))
        self._save(self.optimizer_ref, join(self.save_path, f"optimizer_{suffix}.pt"))
        # if self.lr_scheduler_ref is not None:
        # if self.accelerator_ref is not None:
        # 	self.accelerator_ref.save_model(self.lr_scheduler_ref, join(self.save_path, "scheduler"))
        # else:
        # self._save(self._unwrap(self.lr_scheduler_ref), join(self.save_path, f"scheduler_{suffix}.pt"))
        if self.ema_model_ref is not None:
            self._save(
                self.ema_model_ref, join(self.save_path, f"ema_model_{suffix}.pt")
            )

    def __call__(self, loss):
        self.save_timer += 1

        if self.accelerator_ref is None:
            loss.backward()
        else:
            self.accelerator_ref.backward(loss)
        if self.save_timer % self.grad_accum_steps == 0:
            self.optimizer_ref.step()
            self.optimizer_ref.zero_grad()
        if self.lr_scheduler_ref is not None:
            self.lr_scheduler_ref.step()

        if loss.item() <= self.min_loss:
            self.min_loss = loss.item()
            self._save_everything()

        if self.pbar is not None:
            self.pbar.set_postfix(loss=loss.item())

        if (self.save_timer % self.mandatory_save_period) == 0:
            self._save_everything(f"{self.save_timer}")


class TrainableDiffusionModel:
    """
    Class for training most of my diffusion models
    """

    __slots__ = (
        "model_ref",
        "optimizer_ref",
        "noise_scheduler",
        "criterion",
        "loss_arg_cnt",
        "device",
        "model_type",
        "noise_cov",
        "lr_scheduler_ref",
        "accelerator_ref",
        "EMA_model",
        "EMA_sched",
        "save_path",
        "inv_corr_matrix",
    )

    available_model_types = {"video", "image"}

    def _unwrap(self, obj):
        if self.accelerator_ref is None:
            return obj
        else:
            return self.accelerator_ref.unwrap_model(obj)

    def _save(self, obj, path):
        if self.accelerator_ref is None:
            save(self._unwrap(obj.state_dict()), path)
        else:
            self.accelerator_ref.save(self._unwrap(obj), path)

    def _save_everything(self, suffix="best"):
        if self.accelerator_ref is not None:
            self.accelerator_ref.wait_for_everyone()
        self._save(self.model_ref, join(self.save_path, f"model_{suffix}.pt"))
        self._save(self.optimizer_ref, join(self.save_path, f"optimizer_{suffix}.pt"))
        if self.lr_scheduler_ref is not None:
            self._save(
                self.lr_scheduler_ref, join(self.save_path, f"scheduler_{suffix}.pt")
            )
        if self.EMA_model is not None:
            self._save(self.EMA_model, join(self.save_path, f"ema_model_{suffix}.pt"))

    def _set_noise_cov(self, new_noise_cov):
        if callable(new_noise_cov):
            self.noise_cov = new_noise_cov
        else:
            self.noise_cov = lambda _: new_noise_cov

    def __init__(
        self,
        model_ref,
        optimizer_ref,
        noise_scheduler,
        criterion,
        device="cpu",
        model_type="video",
        noise_cov=lambda x: eye(x),
        lr_scheduler_ref=None,
        accelerator_ref=None,
        EMA_coeff=0.995,
        EMA_start=2500,
    ):

        # basic asserts
        assert (
            model_type in self.available_model_types
        ), "Model type should be 'image' or 'video'."

        # basic fields init
        self.model_ref = model_ref
        self.optimizer_ref = optimizer_ref
        self.noise_scheduler = noise_scheduler
        self.criterion = criterion
        self.loss_arg_cnt = len(Signature.from_callable(criterion).parameters)
        self.device = device
        if accelerator_ref is not None:
            self.device = accelerator_ref.device
        self.model_type = model_type
        self._set_noise_cov(noise_cov)
        self.lr_scheduler_ref = lr_scheduler_ref
        self.accelerator_ref = accelerator_ref
        self.save_path = None
        self.inv_corr_matrix = None

        # EMA setup
        if EMA_start < 0 or EMA_coeff <= 0.0 or EMA_coeff >= 1.0:
            self.EMA_model = None
            self.EMA_sched = None
        else:
            self.EMA_model = (
                deepcopy(self._unwrap(self.model_ref)).eval().requires_grad_(False)
            )
            self.EMA_sched = EMA(beta=EMA_coeff, activation_start=EMA_start)

    def _sample_noise(self, shape):
        match self.model_type:
            case "video":
                noise_gen = NormalVideoNoise(cov_matrix=self.noise_cov(shape[2]))
                return noise_gen.sample(shape).to(self.device)

            case "image":
                return randn(shape, device=self.device, dtype=float32)

    def _train_step(self, batch, labels):
        steps = randint(
            low=0,
            high=len(self.noise_scheduler.timesteps),
            size=(batch.shape[0],),
            device=self.device,
        )
        noise = self._sample_noise(batch.shape)
        noised_videos = self.noise_scheduler.add_noise(batch, noise, steps)
        return (
            noise,
            self.model_ref(
                X=noised_videos,
                t=steps,
                classes=labels,
            ).sample,
        )

    def _one_epoch(self, dataloader, end_processor, class_free_guidance_threshhold):
        pbar = tqdm(dataloader, leave=False)
        end_processor.set_pbar(pbar)
        losses = []
        for i, (batch, labels) in enumerate(pbar):
            batch = batch.to(self.device)
            if self.inv_corr_matrix is None:
                self.inv_corr_matrix = m_inv(self.noise_cov(batch.shape[2])).to(
                    self.device
                )
                self.inv_corr_matrix /= self.inv_corr_matrix.abs().max()

            labels[rand(*labels.shape) < class_free_guidance_threshhold] = -1
            labels = labels.to(self.device)
            noise, predicted_noise = self._train_step(batch, labels)

            if self.loss_arg_cnt == 3:
                loss = self.criterion(noise, predicted_noise, self.inv_corr_matrix)
            else:
                loss = self.criterion(noise, predicted_noise)
            end_processor(loss)
            if self.EMA_model is not None:
                self.EMA_sched.step_ema(self.EMA_model, self._unwrap(self.model_ref))

            losses.append(loss.item())
        return losses

    def fit(
        self,
        dataloader,
        save_path,
        num_epochs=2,
        end_processor=SimpleSaveCallbacker,
        grad_accum_steps=1,
        class_free_guidance_threshhold=0.0,
    ):
        # initial fitting setup
        self.save_path = save_path
        losses = empty(0, len(dataloader))
        end_proc = end_processor(
            model_ref=self.model_ref,
            optimizer_ref=self.optimizer_ref,
            save_path=save_path,
            lr_scheduler_ref=self.lr_scheduler_ref,
            ema_model_ref=self.EMA_model,
            accelerator_ref=self.accelerator_ref,
            grad_accum_steps=grad_accum_steps,
        )

        # Going through epochs
        for _ in tqdm(range(num_epochs)):
            new_losses = self._one_epoch(
                dataloader,
                end_proc,
                class_free_guidance_threshhold=class_free_guidance_threshhold,
            )
            losses = cat([losses, as_tensor(new_losses).unsqueeze(0)], dim=0)

        # saving last versions of models
        self._save_everything(suffix="last")

        return losses

    def load_weights_from(self, other_model, load_to="base_model"):
        available_types = {"base_model", "ema_model"}
        assert (
            load_to in available_types
        ), "You can only load weights into base model or EMA model."

        match load_to:
            case "base_model":
                self.model_ref.requires_grad_(False)
                our_params = dict(self._unwrap(self.model_ref).named_parameters())
            case "ema_model":
                self.EMA_model.requires_grad_(False)
                our_params = dict(self.EMA_model.named_parameters())

        other_params = dict(other_model.named_parameters())

        for param_name, param in our_params.items():
            item = other_params.get(param_name)
            if item is None:
                continue
            param.data = item.data.view(param.data.shape)

        if load_to == "base_model":
            self.model_ref.requires_grad_(True)

    def load_state(
        self,
        base_dir_path,
        suffix="best",
        load_model=True,
        load_optimizer=True,
        load_lr_sched=True,
        load_ema_model=True,
    ):
        if load_model:
            self._unwrap(self.model_ref).load_state_dict(
                load(join(base_dir_path, f"model_{suffix}.pt"), map_location="cpu")
            )
        if load_optimizer:
            self._unwrap(self.optimizer_ref).load_state_dict(
                load(join(base_dir_path, f"optimizer_{suffix}.pt"), map_location="cpu")
            )
        if load_lr_sched:
            self._unwrap(self.lr_scheduler_ref).load_state_dict(
                load(join(base_dir_path, f"scheduler_{suffix}.pt"), map_location="cpu")
            )
        if load_ema_model:
            self.EMA_model.load_state_dict(
                load(join(base_dir_path, f"ema_model_{suffix}.pt"), map_location="cpu")
            )

    def compile(self, mode_model="default", mode_ema="default"):
        set_float32_matmul_precision("high")
        if mode_model is not None:
            self.model_ref = comp(self.model_ref, mode=mode_model)
        if self.EMA_model is not None and mode_ema is not None:
            try:
                _ = self.EMA_model.use_cond
            except:
                self.EMA_model.use_cond = True
            self.EMA_model = comp(self.EMA_model, mode=mode_ema)

    def sample(
        self,
        num_samples,
        prompts=None,
        pic_size=(64, 64),
        num_channels=1,
        num_inference_steps=None,
        video_length=None,
        disable_tqdm=False,
        convert_uint=True,
        override_noise_cov=None,
        override_noise_scheduler=None,
    ):
        assert (
            isinstance(video_length, int) or self.model_type != "video"
        ), "You must specify video_length when generating videos"
        
        if override_noise_scheduler is not None:
            old_scheduler = deepcopy(self.noise_scheduler)
            self.noise_scheduler = override_noise_scheduler

        if override_noise_cov is not None:
            old_noise_cov = self.noise_cov
            self._set_noise_cov(override_noise_cov)
            if hasattr(self.noise_scheduler, "override_noise_cov"):
                self.noise_scheduler.override_noise_cov(override_noise_cov)

        shape = [num_samples, num_channels, *pic_size]
        if self.model_type == "video":
            shape.insert(2, video_length)

        if prompts is not None:
            prompts = prompts.to(self.device)

        if num_inference_steps is not None:
            try:
                old_timesteps = self.noise_scheduler.config.num_train_timesteps
            except:
                old_timesteps = self.noise_scheduler.num_train_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        with no_grad():
            sample = self._sample_noise(shape)

            if self.EMA_model is not None:
                try:
                    _ = self.EMA_model.use_cond
                except:
                    self.EMA_model.use_cond = prompts is not None

                for t in tqdm(self.noise_scheduler.timesteps, disable=disable_tqdm):
                    sample = self.noise_scheduler.scale_model_input(sample, t)
                    residual = self.EMA_model.forward(sample, t, prompts).sample
                    sample = self.noise_scheduler.step(
                        model_output=residual, timestep=t, sample=sample
                    ).prev_sample
            else:
                for t in tqdm(self.noise_scheduler.timesteps, disable=disable_tqdm):
                    sample = self.noise_scheduler.scale_model_input(sample, t)
                    residual = self.model_ref(sample, t, prompts).sample
                    sample = self.noise_scheduler.step(
                        model_output=residual, timestep=t, sample=sample
                    ).prev_sample

        if override_noise_cov is not None:
            self._set_noise_cov(old_noise_cov)
            if hasattr(self.noise_scheduler, "override_noise_cov"):
                self.noise_scheduler.override_noise_cov(old_noise_cov)

        if num_inference_steps is not None:
            self.noise_scheduler.set_timesteps(num_inference_steps=old_timesteps)

        if override_noise_scheduler is not None:
            self.noise_scheduler = old_scheduler

        if convert_uint:
            return ((sample.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).to(uint8)
        else:
            return (sample.detach().cpu().clamp(-1, 1) + 1) / 2
