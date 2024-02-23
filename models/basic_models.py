from torch import sin, arange, zeros
from torch.nn import MSELoss, Embedding, Module
from torch.optim import AdamW
from diffusers import UNet3DConditionModel, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


class MovMNISTModel(Module):
    __slots__ = "use_cond", "cross_att_dim", "model_type", "main_model", "cond_model"

    def __init__(self, sample_size, in_channels, out_channels, layers_per_block,
                 block_out_channels, norm_num_groups, down_block_types,
                 up_block_types, cross_attention_dim,
                 attention_head_dim, model_type="image", class_num=0):
        super().__init__()

        self.use_cond = class_num > 0
        self.cross_att_dim = cross_attention_dim
        self.model_type = model_type
        
        match model_type:
            case "image":
                main_class = UNet2DConditionModel
            case "video":
                main_class = UNet3DConditionModel

        self.main_model = main_class(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

        if self.use_cond:
            self.cond_model = Embedding(
                num_embeddings=class_num,
                embedding_dim=cross_attention_dim,
            )
        else:
            self.cond_model = None

    def _sin_embeddings(self, shape):
        dev = next(self.parameters()).device
        match self.model_type:
            case "video":
                return sin(
                       arange(1, shape[2] + 1, device=dev).view(-1, 1) *\
                       arange(1, self.cross_att_dim + 1, device=dev))\
                       .tile(shape[0]).view(shape[0], shape[2], -1)
            case "image":
                return sin(arange(1, self.cross_att_dim + 1, device=dev))\
                       .tile(shape[0]).view(shape[0], 1, -1)

    def forward(self, X, t, classes=None):
        # Getting embeddings
        if self.use_cond and classes is not None:
            embs = self.cond_model(classes).unsqueeze(1)
        else:
            embs = zeros(X.shape[0], 1, self.cross_att_dim,
                         device=next(self.parameters()).device)

        # Passing through main model
        return self.main_model.forward(sample=X, timestep=t,
                                       encoder_hidden_states=embs)


def init_mov_mnist_model(
    lr_warmup_steps,
    num_epochs,
    beta_start,
    beta_end,
    object_cnt,
    device="cpu",
    total_num_steps=100,
    model_type="image",
    use_labels=False,
    cross_att_dim=4,
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

    model = MovMNISTModel(
        sample_size=(64, 64),
        in_channels=1,
        out_channels=1,
        layers_per_block=3,
        block_out_channels=(
            32,
            48,
            64
            ),
        norm_num_groups=8,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        cross_attention_dim=cross_att_dim,
        attention_head_dim=cross_att_dim,
        model_type=model_type,
        class_num=55 if use_labels else 0,
    )
    model.to(device)
    model.train()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=total_num_steps, beta_start=beta_start,
        beta_end=beta_end)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(object_cnt * num_epochs),
    )

    criterion = MSELoss()

    output = (model, noise_scheduler, optimizer, lr_scheduler, criterion)

    return output 
