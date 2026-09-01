from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ct2pet_dataset import CTPETDataset
from unet3d import Unet3D
from conditional_gaussian_diffusion import ConditionalGaussianDiffusion

DATA_ROOT = Path("/workspace/data/hecktor_numpy_seed42")
TRAIN_ROOT = DATA_ROOT / "train"

OUT_DIR = Path(
    "/workspace/diffusion_ct2pet/"
    "baseline_conditional_diffusion_50k"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PET_CLIP = 30.0

CT_MIN = -1024.0
CT_MAX = 1500.0

NUM_FRAMES = 7
IMAGE_SIZE = 96

UNET_DIM = 32
UNET_DIM_MULTS = (1, 2, 4)

ATTN_HEADS = 4
ATTN_DIM_HEAD = 16

DIFFUSION_TIMESTEPS = 1000
LOSS_TYPE = "l1"

BATCH_SIZE = 2
LEARNING_RATE = 1e-4

TOTAL_STEPS = 50_000
SEED = 42

LOG_EVERY = 50
SAVE_EVERY = 10_000

RESUME_CKPT = None


def normalize_pet(pet):
    return (
        pet.clamp(
            0.0,
            PET_CLIP,
        )
        / PET_CLIP
    )


def normalize_ct(ct):
    ct = ct.clamp(
        CT_MIN,
        CT_MAX,
    )

    return (
        2.0
        * (ct - CT_MIN)
        / (CT_MAX - CT_MIN)
        - 1.0
    )



def build_dataset():
    return CTPETDataset(
        ct_dir=str(
            TRAIN_ROOT / "trainA"
        ),
        pet_dir=str(
            TRAIN_ROOT / "trainB"
        ),
        label_dir=str(
            TRAIN_ROOT / "trainLabel"
        ),
        expected_shape=(
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        strict=True,
    )

def build_model(device):

    model = Unet3D(
        dim=UNET_DIM,
        channels=2,
        out_dim=1,
        dim_mults=UNET_DIM_MULTS,
        attn_heads=ATTN_HEADS,
        attn_dim_head=ATTN_DIM_HEAD,
    ).to(device)

    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        num_frames=NUM_FRAMES,
        channels=1,
        timesteps=DIFFUSION_TIMESTEPS,
        loss_type=LOSS_TYPE,
    ).to(device)

    return model, diffusion

def compute_noise_loss(
    diffusion,
    x_start,
    ct,
    t,
    noise,
):
    x_noisy = diffusion.q_sample(
        x_start=x_start,
        t=t,
        noise=noise,
    )

    model_input = diffusion._model_input(
        x_noisy,
        ct,
    )

    noise_pred = diffusion.denoise_fn(
        model_input,
        t,
    )

    if noise_pred.shape != noise.shape:
        raise RuntimeError(
            f"Wrong baseline output shape: "
            f"{tuple(noise_pred.shape)} vs noise {tuple(noise.shape)}"
        )

    if LOSS_TYPE == "l1":
        loss = F.l1_loss(
            noise_pred,
            noise,
        )
    elif LOSS_TYPE == "l2":
        loss = F.mse_loss(
            noise_pred,
            noise,
        )
    else:
        raise ValueError(
            f"Unsupported LOSS_TYPE={LOSS_TYPE}"
        )

    return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ds = build_dataset()

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True,
    )

    model, diffusion = build_model(
        device
    )

    with torch.no_grad():
        test_x = torch.randn(
            1,
            2,
            NUM_FRAMES,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
        )

        test_t = torch.randint(
            0,
            DIFFUSION_TIMESTEPS,
            (1,),
            device=device,
        )

        test_out = model(
            test_x,
            test_t,
        )

    expected_shape = (
        1,
        1,
        NUM_FRAMES,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    print(
        f"Model output shape        : {tuple(test_out.shape)}"
    )

    if test_out.shape != expected_shape:
        raise RuntimeError(
            f"Wrong baseline output shape: "
            f"{tuple(test_out.shape)}; expected {expected_shape}"
        )

    if not torch.isfinite(
        test_out
    ).all():
        raise RuntimeError(
            "Non-finite output in baseline architecture smoke test."
        )

    print(
        "Baseline shape check      : PASS",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    start_step = 1

    if RESUME_CKPT is not None:
        resume_path = Path(
            RESUME_CKPT
        )

        print(
            f"Resuming from checkpoint: {resume_path}",
            flush=True,
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        start_step = int(
            checkpoint.get(
                "step",
                0,
            )
        ) + 1


    iterator = iter(
        loader
    )

    history = []

    start = time.time()

    print(
        "\nTraining starts...\n",
        flush=True,
    )

    for step in range(
        start_step,
        TOTAL_STEPS + 1,
    ):
        try:
            batch = next(
                iterator
            )

        except StopIteration:
            iterator = iter(
                loader
            )

            batch = next(
                iterator
            )

        ct = normalize_ct(
            batch[
                "ct"
            ]
        ).to(
            device,
            non_blocking=True,
        )

        pet01 = normalize_pet(
            batch[
                "pet"
            ]
        ).to(
            device,
            non_blocking=True,
        )

        x_start = (
            pet01
            * 2.0
            - 1.0
        )

        b = x_start.shape[0]

        t = torch.randint(
            0,
            DIFFUSION_TIMESTEPS,
            (b,),
            device=device,
        ).long()

        noise = torch.randn_like(
            x_start
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        noise_loss = compute_noise_loss(
            diffusion,
            x_start,
            ct,
            t,
            noise,
        )

        if not torch.isfinite(
            noise_loss
        ):
            raise RuntimeError(
                f"Non-finite baseline loss at step {step}"
            )

        noise_loss.backward()

        for p in model.parameters():
            if (
                p.grad is not None
                and not torch.isfinite(
                    p.grad
                ).all()
            ):
                raise RuntimeError(
                    f"Non-finite gradient at step {step}"
                )

        optimizer.step()

        history.append(
            float(
                noise_loss
                .detach()
                .item()
            )
        )

        if (
            step == 1
            or step % LOG_EVERY == 0
        ):
            window = min(
                LOG_EVERY,
                len(history),
            )

            recent = float(
                np.mean(
                    history[
                        -window:
                    ]
                )
            )

            print(
                f"step {step:06d}/{TOTAL_STEPS} | "
                f"noise_loss={recent:.6f}",
                flush=True,
            )

        if (
            step % SAVE_EVERY == 0
            or step == TOTAL_STEPS
        ):
            checkpoint_data = {
                "step":
                    step,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "config": {
                    "architecture":
                        "vanilla_ct_conditional_unet3d",

                    "pet_clip":
                        PET_CLIP,

                    "ct_min":
                        CT_MIN,

                    "ct_max":
                        CT_MAX,

                    "learning_rate":
                        LEARNING_RATE,

                    "batch_size":
                        BATCH_SIZE,

                    "total_steps":
                        TOTAL_STEPS,

                    "diffusion_timesteps":
                        DIFFUSION_TIMESTEPS,

                    "seed":
                        SEED,
                },
            }

            periodic_ckpt = (
                OUT_DIR
                / f"checkpoint_step_{step:06d}.pt"
            )

            torch.save(
                checkpoint_data,
                periodic_ckpt,
            )

            torch.save(
                checkpoint_data,
                OUT_DIR
                / "latest_checkpoint.pt",
            )

            print(
                f"Checkpoint saved: {periodic_ckpt}",
                flush=True,
            )

    elapsed = (
        time.time()
        - start
    )

    def avg_first(n=20):
        return float(
            np.mean(
                history[
                    :min(
                        n,
                        len(history),
                    )
                ]
            )
        )

    def avg_last(n=20):
        return float(
            np.mean(
                history[
                    -min(
                        n,
                        len(history),
                    ):
                ]
            )
        )

    print()
    print("=" * 88)
    print("BASELINE 50K RESULT")
    print("=" * 88)

    print(
        f"Noise loss | "
        f"first20={avg_first():.6f} | "
        f"last20={avg_last():.6f}"
    )

    print(
        f"Runtime: {elapsed:.2f} sec"
    )

    final_ckpt = (
        OUT_DIR
        / "final_checkpoint.pt"
    )

    torch.save(
        {
            "step":
                TOTAL_STEPS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "config": {
                "architecture":
                    "vanilla_ct_conditional_unet3d",

                "pet_clip":
                    PET_CLIP,

                "ct_min":
                    CT_MIN,

                "ct_max":
                    CT_MAX,

                "learning_rate":
                    LEARNING_RATE,

                "batch_size":
                    BATCH_SIZE,

                "total_steps":
                    TOTAL_STEPS,

                "diffusion_timesteps":
                    DIFFUSION_TIMESTEPS,

                "seed":
                    SEED,
            },
        },
        final_ckpt,
    )

    print(
        f"Saved checkpoint: {final_ckpt}"
    )

if __name__ == "__main__":
    main()