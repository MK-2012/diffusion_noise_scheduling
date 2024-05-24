from torch import sin, arange, zeros, eye
from torch.nn import Embedding, Module, MSELoss
from torch.optim import AdamW
from diffusers import UNet3DConditionModel, UNet2DConditionModel, DDIMScheduler, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.custom_ddpm_scheduler import *
from utils.noise_gen import *
from utils.training import *
from utils.utils import video_loss


class MovMNISTModel(Module):
    __slots__ = "use_cond", "cross_att_dim", "model_type", "need_time_embs", "main_model", "cond_model"

    def __init__(self, model_kwargs, model_type="image", class_num=0, model_type_override=None):
        super().__init__()

        self.use_cond = class_num > 0
        self.cross_att_dim = model_kwargs["cross_attention_dim"]
        self.model_type = model_type

        if model_type_override is None:
            match model_type:
                case "image":
                    main_class = UNet2DConditionModel
                case "video":
                    main_class = UNet3DConditionModel
            self.need_time_embs = False
        else:
            main_class = model_type_override
            self.need_time_embs = True

        self.main_model = main_class(
            **model_kwargs,
        )

        if self.use_cond:
            self.cond_model = Embedding(
                num_embeddings=class_num,
                embedding_dim=self.cross_att_dim,
            )
        else:
            self.cond_model = None

    def forward(self, X, t, classes=None):
        # Getting embeddings
        if self.use_cond and classes is not None:
            model_inds = classes != -1
            embs = zeros(X.shape[0], 1, self.cross_att_dim, device=next(self.parameters()).device)
            embs[model_inds] = self.cond_model(classes[model_inds]).unsqueeze(1)
        else:
            embs = zeros(X.shape[0], 1, self.cross_att_dim,
                         device=next(self.parameters()).device)

        # Passing through main model
        if not self.need_time_embs:
            return self.main_model.forward(sample=X, timestep=t,
                                       encoder_hidden_states=embs)
        else:
            return self.main_model.forward(sample=X.swapaxes(1, 2), timestep=t,
                                       encoder_hidden_states=embs, added_time_ids=torch.ones(X.shape[0], device=next(self.parameters()).device))


def init_mov_mnist_model(
    lr_warmup_steps,
    num_epochs,
    beta_start,
    beta_end,
    object_cnt,
    noise_cov_matrix,
    noise_scheduler=CustomDDPMScheduler,
    device="cpu",
    total_num_steps=100,
    model_type="image",
    use_labels=False,
    cross_att_dim=4,
    EMA_start=1200,
    EMA_coeff=0.995,
    criterion=None,
):
    match model_type:
        case "image":
            down_blocks = (
                "CrossAttnDownBlock2D",
                # "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            )
            up_blocks = (
                "CrossAttnUpBlock2D",
                # "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                # "CrossAttnUpBlock2D",
            )
        case "video":
            down_blocks = (
                "CrossAttnDownBlock3D",
                # "DownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
            )
            up_blocks = (
                "CrossAttnUpBlock3D",
                # "UpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
                # "CrossAttnUpBlock3D",
            )
        case _:
            raise ValueError(f"Unknown model type found: {model_type}")

    # Model creation
    model_kwargs = {
        "sample_size": (64, 64),
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 3,
        "block_out_channels": (
            32,
            48,
            64
        ),
        "norm_num_groups": 8,
        "down_block_types": down_blocks,
        "up_block_types": up_blocks,
        "cross_attention_dim": cross_att_dim,
        "attention_head_dim": cross_att_dim,
    }
    model = MovMNISTModel(
        model_kwargs=model_kwargs,
        model_type=model_type,
        class_num=55 if use_labels else 0,
    )
    model.to(device)
    model.train()

    # Noise scheduler
    noise_scheduler = noise_scheduler(
        num_train_timesteps=total_num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_cov_matrix=noise_cov_matrix,
        timestep_spacing="trailing",
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    #Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(object_cnt * num_epochs),
    )

    if criterion is None:
        match model_type:
            case "image":
                criterion = MSELoss()
            case "video":
                criterion = video_loss
            case _:
                raise ValueError(f"Unknown model type found during criterion deduction: {model_type}")

    # Smart model initialization
    SmartModel = TrainableDiffusionModel(
        model_ref=model,
        optimizer_ref=optimizer,
        lr_scheduler_ref=lr_scheduler,
        noise_scheduler=noise_scheduler,
        criterion=criterion,
        device=device,
        model_type=model_type,
        EMA_start=EMA_start,
        EMA_coeff=EMA_coeff,
        noise_cov=noise_cov_matrix,
    )

    return SmartModel


def init_big_mov_mnist_model(
    lr_warmup_steps,
    num_epochs,
    beta_start,
    beta_end,
    object_cnt,
    noise_cov_matrix,
    noise_scheduler=CustomDDPMScheduler,
    device="cpu",
    total_num_steps=100,
    model_type="image",
    use_labels=False,
    cross_att_dim=4,
    EMA_start=1200,
    EMA_coeff=0.995,
    criterion=None,
):
    match model_type:
        case "image":
            down_blocks = (
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            )
            up_blocks = (
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            )
            model_kwargs = {
                "sample_size": (64, 64),
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 3,
                "block_out_channels": (
                    32,
                    64,
                    96,
                    192,
                ),
                "norm_num_groups": 8,
                "down_block_types": down_blocks,
                "up_block_types": up_blocks,
                "cross_attention_dim": cross_att_dim,
                "attention_head_dim": cross_att_dim,
            }

        case "video":
            # down_blocks = (
            #     "CrossAttnDownBlockSpatioTemporal",
            #     "DownBlockSpatioTemporal",
            #     "DownBlockSpatioTemporal",
            #     "DownBlockSpatioTemporal",
            # )
            # up_blocks = (
            #     "CrossAttnUpBlockSpatioTemporal",
            #     "UpBlockSpatioTemporal",
            #     "UpBlockSpatioTemporal",
            #     "UpBlockSpatioTemporal",
            # )

            # model_kwargs = {
            #     "sample_size": (64, 64),
            #     "in_channels": 1,
            #     "out_channels": 1,
            #     "layers_per_block": 2,
            #     "block_out_channels": (
            #         32,
            #         64,
            #         96,
            #         192,
            #     ),
            #     "down_block_types": down_blocks,
            #     "up_block_types": up_blocks,
            #     "addition_time_embed_dim": 4,
            #     "projection_class_embeddings_input_dim": cross_att_dim,
            #     "cross_attention_dim": cross_att_dim,
            #     "num_frames": 20,
            #     "num_attention_heads": [
            #         1,
            #         2,
            #         4,
            #         4,
            #     ],
            # }


            down_blocks = (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
            )
            up_blocks = (
                "CrossAttnUpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
            )

            model_kwargs = {
                "sample_size": (64, 64),
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 3,
                "block_out_channels": (
                    32,
                    64,
                    96,
                    192,
                ),
                "norm_num_groups": 8,
                "down_block_types": down_blocks,
                "up_block_types": up_blocks,
                "cross_attention_dim": cross_att_dim,
                "attention_head_dim": cross_att_dim,
            }
        case _:
            raise ValueError(f"Unknown model type found: {model_type}")

    # Model creation
    model = MovMNISTModel(
        model_kwargs=model_kwargs,
        model_type=model_type,
        class_num=55 if use_labels else 0,
        # model_type_override=UNetSpatioTemporalConditionModel,
    )
    model.to(device)
    model.train()

    # Noise scheduler
    noise_scheduler = noise_scheduler(
        num_train_timesteps=total_num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        noise_cov_matrix=noise_cov_matrix,
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    #Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(object_cnt * num_epochs),
    )

    # Criterion
    if criterion is None:
        match model_type:
            case "image":
                criterion = MSELoss()
            case "video":
                criterion = video_loss
            case _:
                raise ValueError(f"Unknown model type found during criterion deduction: {model_type}")

    # Smart model initialization
    SmartModel = TrainableDiffusionModel(
        model_ref=model,
        optimizer_ref=optimizer,
        lr_scheduler_ref=lr_scheduler,
        noise_scheduler=noise_scheduler,
        criterion=criterion,
        device=device,
        model_type=model_type,
        EMA_start=EMA_start,
        EMA_coeff=EMA_coeff,
        noise_cov=noise_cov_matrix,
    )

    return SmartModel
