import torch
import torch.nn.functional as F
from einops import rearrange

from gaussian_diffusion import GaussianDiffusion


class ConditionalGaussianDiffusion(GaussianDiffusion):

    def _check_ct(self, x_pet, ct):
        if ct is None:
            raise ValueError("CT is not provided.")

        if ct.ndim != 5:
            raise ValueError(
                f"CT must follow this structure:[B,1,F,H,W]"
            )

        if ct.shape[0] != x_pet.shape[0]:
            raise ValueError(
                f"Batch mismatch"
            )

        if ct.shape[2:] != x_pet.shape[2:]:
            raise ValueError(
                f"CT/PET size missmatch"
            )

        if ct.shape[1] != 1:
            raise ValueError(
                f"Step 3 expects 1 CT channel"
            )

    def _model_input(self, x_pet, ct):
        self._check_ct(x_pet, ct)
        return torch.cat((x_pet, ct), dim=1)

    def p_losses(self, x_start, t, ct=None, noise=None, **kwargs):
        noise = torch.randn_like(x_start) if noise is None else noise

        # Forward diffusion is applied to PET.
        x_noisy = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise
        )

        # U-Net receives noisy PET + CT.
        model_input = self._model_input(x_noisy, ct)

        # U-Net predicts only PET noise
        noise_pred = self.denoise_fn(
            model_input,
            t,
            **kwargs
        )

        if noise_pred.shape != noise.shape:
            raise ValueError(
                f"Noise prediction shape mismatch: "
                f"pred={tuple(noise_pred.shape)}, target={tuple(noise.shape)}"
            )

        if self.loss_type == "l1":
            loss = F.l1_loss(noise, noise_pred)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, noise_pred)
        else:
            raise NotImplementedError()

        return loss

    def p_mean_variance(self, x, t, clip_denoised: bool, ct=None):
        model_input = self._model_input(x, ct)

        noise_pred = self.denoise_fn(
            model_input,
            t
        )

        x_recon = self.predict_start_from_noise(
            x,
            t=t,
            noise=noise_pred
        )

        if clip_denoised:
            s = 1.0

            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, "b ... -> b (...)").abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )
                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s

        return self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t
        )

    @torch.inference_mode()
    def p_sample(self, x, t, ct=None, clip_denoised=True):

        b = x.shape[0]

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            ct=ct
        )

        noise = torch.randn_like(x)

        nonzero_mask = (
            1 - (t == 0).float()
        ).reshape(
            b,
            *((1,) * (len(x.shape) - 1))
        )

        return (
            model_mean
            + nonzero_mask
            * (0.5 * model_log_variance).exp()
            * noise
        )

    @torch.inference_mode()
    def p_sample_loop(self, shape, ct):

        device = self.betas.device
        b = shape[0]

        self._check_ct(
            torch.empty(shape, device=device),
            ct
        )

        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full(
                (b,),
                i,
                device=device,
                dtype=torch.long
            )

            img = self.p_sample(
                img,
                t,
                ct=ct
            )

        from video_dataset import unnormalize_img
        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, ct):
    
        if ct is None:
            raise ValueError("CT condition must be provided for sampling.")

        batch_size = ct.shape[0]
        image_size = self.image_size

        shape = (
            batch_size,
            self.channels,      
            self.num_frames,
            image_size,
            image_size
        )

        return self.p_sample_loop(
            shape,
            ct=ct
        )