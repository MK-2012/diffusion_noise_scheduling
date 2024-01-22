from torch.nn import MSELoss
from torch.optim import AdamW
from diffusers import UNet3DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

def init_basic_model(
    lr_warmup_steps,
    num_epochs,
    beta_start,
    beta_end,
    object_cnt,
    device="cpu",
    total_num_steps=1000,
):
    model = UNet3DConditionModel(
        sample_size=(240, 320),
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(12,),
        norm_num_groups=2,
        down_block_types=(
#             "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
#             "CrossAttnUpBlock3D",
            "UpBlock3D",
          ),
        cross_attention_dim=24,
        attention_head_dim=8,
    )
    model.to(device)
    model.train()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=total_num_steps, beta_start=beta_start, beta_end=beta_end)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(object_cnt * num_epochs),
    )

    criterion = MSELoss()

    output = (model, noise_scheduler, optimizer, lr_scheduler, criterion)

    return output 
