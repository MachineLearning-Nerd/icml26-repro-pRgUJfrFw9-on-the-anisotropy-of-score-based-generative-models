import utils.improved_diffusion as improved_diffusion
import utils.dit as dit
import torch as th
from typing import Any, Dict, List, Optional


class Diffuser(th.nn.Module):
    """
    VP diffuser module.
    """

    def __init__(self,
                 shape: List[int],
                 T: int,
                 linear: bool,
                 model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.dims = len(shape) - 1
        if "channel_mults" in model_cfg:
            self.model = improved_diffusion.UNetModel(in_channels=shape[0],
                                                      out_channels=shape[0],
                                                      channel_mult=[1,] + model_cfg["channel_mults"],
                                                      model_channels=model_cfg["base_channels"],
                                                      num_res_blocks=model_cfg["num_res_attn_blocks"],
                                                      attention_resolutions=[2 ** i for i, is_attn in enumerate(model_cfg["is_attn"]) if is_attn],  # noqa: E501
                                                      dropout=model_cfg["dropout"],
                                                      num_heads=model_cfg["num_heads"],
                                                      use_scale_shift_norm=model_cfg["use_scale_shift_norm"],
                                                      conv_resample=model_cfg.get("conv_resample", True),
                                                      interpolation=model_cfg.get("interpolation", "nearest"),
                                                      downsample_with_pool=model_cfg.get("downsample_with_pool", True),
                                                      skip_connections=model_cfg.get("skip_connections", True),
                                                      padding_mode=model_cfg.get("padding_mode", "zeros"),
                                                      dims=self.dims,
                                                      )
        else:
            self.model = dit.DiT(input_size=shape[1],
                                 patch_size=model_cfg["patch_size"],
                                 in_channels=shape[0],
                                 hidden_size=model_cfg["hidden_size"],
                                 depth=model_cfg["depth"],
                                 num_heads=model_cfg["num_heads"],
                                 mlp_ratio=model_cfg["mlp_ratio"],
                                 )
        if linear:
            betas = th.linspace(0.1 / T, 20 / T, T, dtype=th.float64)
        else:
            s = 0.008
            steps = th.linspace(0., T, T + 1, dtype=th.float64)
            ft = th.cos(((steps / T + s) / (1 + s)) * th.pi * 0.5) ** 2
            betas = th.clip(1 - ft[1:] / ft[:T], 0., 0.999)

        sqrt_betas = th.sqrt(betas)
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        one_minus_alphas_cumprod = 1 - alphas_cumprod
        sqrt_alphas = th.sqrt(alphas)

        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = th.sqrt(one_minus_alphas_cumprod)

        self.register_buffer("betas", betas.to(th.float32))
        self.register_buffer("sqrt_betas", sqrt_betas.to(th.float32))
        self.register_buffer("alphas", alphas.to(th.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(th.float32))
        self.register_buffer("one_minus_alphas_cumprod", one_minus_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_alphas", sqrt_alphas.to(th.float32))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(th.float32))

        T = th.tensor(T, dtype=th.float32).unsqueeze_(0)
        self.register_buffer("T", T)

    def forward(self, x_start: th.Tensor, start: Optional[th.Tensor] = None) -> th.Tensor:
        """
        DDP
        """
        if start is None:
            start = th.tensor(x_start.shape[0] * [-1],
                              dtype=int,
                              device=x_start.device).reshape(-1, *([1] * (x_start.ndim - 1)))

        end = self.randint(batch_size=x_start.shape[0], device=x_start.device, low=start + 1)
        x_end = self.noise(x_start, end, start=start)
        mask = (start >= 0).to(x_start.dtype).view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_alpha_cumprod_s = self.sqrt_alphas_cumprod[start] * mask + (1 - mask)
        sqrt_alpha_cumprod_se = self.sqrt_alphas_cumprod[end] / sqrt_alpha_cumprod_s
        eps = (x_end - sqrt_alpha_cumprod_se * x_start) / self.sqrt_one_minus_alphas_cumprod[end]
        return th.nn.functional.mse_loss(self.epsilon(x_end, end), eps)

    def noise(self, x: th.Tensor, end: th.Tensor, start: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Noising from start to end.
        For negative values, keep x unchanged.
        """

        if start is None:
            start = th.full_like(input=end, fill_value=-1)

        mask_e = (end >= 0).to(x.dtype).view(-1, *([1] * (x.ndim - 1)))
        end_clamped = end.clamp(min=0)

        mask_s = (start >= 0).to(x.dtype).view(-1, *([1] * (x.ndim - 1)))
        alpha_cumprod_s = self.alphas_cumprod[start] * mask_s + (1 - mask_s)

        alpha_cumprod = self.alphas_cumprod[end_clamped] / alpha_cumprod_s
        sqrt_alpha_cumprod = alpha_cumprod.sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod).sqrt()
        eps = th.randn_like(x)

        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * eps
        return mask_e * noisy_x + (1 - mask_e) * x

    def randint(self,
                batch_size: int,
                device: th.device,
                low: th.Tensor = 0,
                high: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Sample a random time step.
        """
        if high is None:
            high = len(self.betas)

        rand_uniform = th.rand([batch_size, 1] + self.dims * [1], device=device)
        return low + (rand_uniform * (high - low)).floor().to(th.int64)

    def epsilon(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        return self.model(x, t * 1000. / len(self.betas))

    @th.inference_mode()
    def ddim(diffuser: th.nn.Module,
             init: th.Tensor,
             condition: Optional[th.Tensor] = None,
             steps: Optional[int] = None,
             eta: float = 1.,
             clamp_min: float = -th.inf,
             clamp_max: float = th.inf) -> th.Tensor:
        """
        Diffuse a tensor under a DDIM schedule.
        Defaults to DDPM for steps=None, eta=1.

        Args:
            diffuser: contains the non-blind denoiser (epsilon prediction).
                Should take ([x, condition], t) as input if conditional and (x, t) otherwise.
            init: the initial tensor.
            condition: the condition tensor. None if the denoiser is not conditional.
            steps: the number of steps to diffuse.
            eta: the eta parameter.
            clamp_min: the minimum value of the clamp.
            clamp_max: the maximum value of the clamp.
        """

        x = init
        if not steps:
            steps = int(diffuser.T)
        times = th.linspace(int(diffuser.T) - 1, 0, steps, dtype=int, device=x.device)

        for index in range(len(times)):
            t = times[index]
            sqrt_alpha_cumprod = diffuser.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod = diffuser.sqrt_one_minus_alphas_cumprod[t]

            inpt = th.cat((x, condition), 1) if condition is not None else x
            eps_pred = diffuser.epsilon(inpt, t)
            x0_pred = (x - sqrt_one_minus_alpha_cumprod * eps_pred) / sqrt_alpha_cumprod
            x0_pred.clamp_(clamp_min, clamp_max)
            eps_pred = (x - sqrt_alpha_cumprod * x0_pred) / sqrt_one_minus_alpha_cumprod

            if index == len(times) - 1:
                return x0_pred

            prev_t = times[index + 1]
            alpha_cumprod = diffuser.alphas_cumprod[t]
            alpha_cumprod_prev = diffuser.alphas_cumprod[prev_t]
            sqrt_alpha_cumprod_prev = diffuser.sqrt_alphas_cumprod[prev_t]
            sqrt_one_minus_alpha_cumprod_prev = diffuser.sqrt_one_minus_alphas_cumprod[prev_t]

            std = eta * (sqrt_one_minus_alpha_cumprod_prev / sqrt_one_minus_alpha_cumprod) * th.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)  # noqa: E501
            x = sqrt_alpha_cumprod_prev * x0_pred + th.sqrt(1 - alpha_cumprod_prev - std ** 2) * eps_pred + std * th.randn_like(x)  # noqa: E501

    @th.inference_mode()
    def sample(self,
               init: th.Tensor,
               steps: Optional[int] = None,
               eta: float = 1.,
               clamp_min: float = -th.inf,
               clamp_max: float = th.inf
               ) -> th.Tensor:
        """
        Sample from the model.
        """

        out = self.ddim(init=init,
                        steps=steps,
                        eta=eta,
                        clamp_min=clamp_min,
                        clamp_max=clamp_max)
        return out
