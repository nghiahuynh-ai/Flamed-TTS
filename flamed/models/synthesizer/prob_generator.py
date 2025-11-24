import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = Block1D(dim, dim_out, groups=groups)

    def forward(self, x, mask):
        h = self.block(x, mask)
        return x + h


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 2-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, :, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, kernel=31, stride=1, padding=15, expand=1, groups=None):
        super().__init__()
        if groups == None:
            groups = channels

        self.conv_1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.ln_1 = torch.nn.GroupNorm(channels, channels)

        self.conv_2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * expand,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv_3 = nn.Conv1d(
            in_channels=channels * expand,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = x.transpose(1, -1)
        x = x + self.conv_3(F.gelu(self.conv_2(self.ln_1(self.conv_1(x)))))
        x = x.transpose(1, -1)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels,
        convnext_kernel=31,
        convnext_stride=1,
        convnext_padding=15,
        convnext_expand=1,
        convnext_groups=None,
    ):
        super().__init__()
        self.channels = channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels, bias=True)
        )

        self.ln_conv = nn.LayerNorm(channels, eps=1e-6)
        self.conv_in = ConvNeXtBlock(
            channels,
            kernel=convnext_kernel,
            stride=convnext_stride,
            padding=convnext_padding,
            expand=convnext_expand,
            groups=convnext_groups,
        )

        self.ln_mlp = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x, y):
        (
            shift_conv, 
            scale_conv, 
            gate_conv, 
            shift_mlp, 
            scale_mlp, 
            gate_mlp
        ) = self.adaLN_modulation(y).chunk(6, dim=-1)
        x = x + gate_conv * self.conv_in(modulate(self.ln_conv(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.ln_mlp(x), shift_mlp, scale_mlp))
        return x
    

class ConditionDownSampler(nn.Module):

    def __init__(self, in_channel, out_channel, n_stages=1, n_groups=8):
        super().__init__()

        self.n_stages = n_stages
        self.resblocks = nn.ModuleList()
        self.downblocks = nn.ModuleList()

        for _ in range(n_stages):
            self.resblocks.append(
                ResnetBlock1D(dim=in_channel, dim_out=in_channel)
            )
            self.downblocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=in_channel // 2,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(n_groups, in_channel // 2),
                    nn.ReLU(),
                )
            )
            in_channel = in_channel // 2

        self.proj_out = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU()
        )

    def forward(self, x, mask):
        mask = mask.transpose(1, -1)
        x = x.transpose(1, -1)
        for ith in range(self.n_stages):
            x = self.resblocks[ith](x, mask)
            x = self.downblocks[ith](x)
        x = x.transpose(1, -1)
        return self.proj_out(x)


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(
            self, 
            model_channels, 
            out_channels,
            convnext_kernel,
            convnext_stride,
            convnext_padding,
            convnext_expand,
            convnext_groups,
        ):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 5 * model_channels, bias=True)
        )

        self.norm_in = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.conv_in = ConvNeXtBlock(
            model_channels,
            kernel=convnext_kernel,
            stride=convnext_stride,
            padding=convnext_padding,
            expand=convnext_expand,
            groups=convnext_groups,
        )

        self.norm_out = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.conv_out = nn.Conv1d(
            in_channels=model_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, c):
        (
            shift_conv, 
            scale_conv, 
            gate_conv, 
            shift_mlp, 
            scale_mlp
        ) = self.adaLN_modulation(c).chunk(5, dim=-1)

        x = x + gate_conv * self.conv_in(modulate(self.norm_in(x), shift_conv, scale_conv))

        x = modulate(self.norm_out(x), shift_mlp, scale_mlp)
        x = x.transpose(1, -1)
        x = self.conv_out(x)
        x = x.transpose(1, -1)

        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_dim: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        spk_dim,
        num_res_blocks,
        convnext_kernel,
        convnext_stride,
        convnext_padding,
        convnext_expand,
        convnext_groups,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(spk_dim, model_channels)
        self.proj_in = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                    convnext_kernel,
                    convnext_stride,
                    convnext_padding,
                    convnext_expand,
                    convnext_groups,
                ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(
            model_channels, 
            out_channels,
            convnext_kernel,
            convnext_stride,
            convnext_padding,
            convnext_expand,
            convnext_groups,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv_out.weight, 0)
        nn.init.constant_(self.final_layer.conv_out.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c.unsqueeze(1)

        x = self.proj_in(x)
        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
    

class QuantizerEncoding(nn.Module):
    def __init__(self, n_quantizers, hidden_dim):
        super(QuantizerEncoding, self).__init__()
        
        self.quantizer_ids = torch.arange(n_quantizers).expand((1, -1))
        self.quantizer_emb = nn.Embedding(n_quantizers, hidden_dim)
        
    def forward(self, x):
        identifier = self.quantizer_emb(self.quantizer_ids.to(x.device))
        x = x + identifier.unsqueeze(2).expand(-1, -1, x.size(2), -1).contiguous()
        b, q, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, q * d)
        return x


class ProbGenerator(nn.Module):
    """Diffusion Loss"""
    def __init__(self, config):
        super(ProbGenerator, self).__init__()

        self.target_dim = config['target_dim']
        self.sigma_min = config['sigma_min']

        self.quantizer_encoding = QuantizerEncoding(
            n_quantizers=config['n_quantizers'],
            hidden_dim=config['cond_dim']
        )
        self.cond_downsampling = ConditionDownSampler(
            in_channel=config['n_quantizers'] * config['cond_dim'],
            out_channel=config['target_dim'],
            n_stages=config['downsampling_stages'],
        )
        self.denoiser = SimpleMLPAdaLN(
            in_channels=config['target_dim'],
            model_channels=config['hidden_dim'],
            out_channels=config['target_dim'],
            spk_dim=config['spk_dim'],
            num_res_blocks=config['n_layers'],
            convnext_kernel=config['convnext']['kernel_size'],
            convnext_stride=config['convnext']['stride'],
            convnext_padding=config['convnext']['padding'],
            convnext_expand=config['convnext']['expand'],
            convnext_groups=config['convnext']['groups'],
        )
        self.cfg_dropout_prob = float(config.get('cfg_dropout_prob', 0.0))
        self.guidance_scale = float(config.get('guidance_scale', 1.0))

    def _apply_cfg_dropout(self, spk):
        if not self.training or self.cfg_dropout_prob <= 0:
            return spk
        drop_mask = torch.rand(
            spk.size(0), 1, device=spk.device
        ) < self.cfg_dropout_prob
        if not drop_mask.any():
            return spk
        drop_mask = drop_mask.view(spk.size(0), *([1] * (spk.ndim - 1)))
        fake_spk = torch.randn_like(spk)
        return torch.where(drop_mask, fake_spk, spk)

    def _expand_timesteps(self, t, batch_size):
        if t.size(0) == batch_size:
            return t
        expanded_shape = (batch_size,) + tuple(t.shape[1:])
        return t.expand(*expanded_shape)

    def _guided_velocity(
        self,
        xt,
        t,
        spk_cond,
        spk_uncond=None,
        mask=None,
        guidance_scale=None,
    ):
        batch_size = xt.size(0)
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        else:
            guidance_scale = float(guidance_scale)

        t = self._expand_timesteps(t, batch_size)
        use_guidance = (
            spk_uncond is not None and not math.isclose(guidance_scale, 1.0)
        )
        if use_guidance:
            xt_cat = torch.cat([xt, xt], dim=0)
            t_cat = torch.cat([t, t], dim=0)
            spk_cat = torch.cat([spk_uncond, spk_cond], dim=0)
            vt_uncond, vt_cond = self.denoiser(xt_cat, t_cat, spk_cat).chunk(2, dim=0)
            vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)
        else:
            vt = self.denoiser(xt, t, spk_cond)

        if mask is not None:
            vt = vt * mask.to(vt.dtype)

        return vt

    def compute_loss(self, x1, cond, spk, mask):
        cond = self.quantizer_encoding(cond)
        cond = self.cond_downsampling(cond, mask)

        t = torch.rand((cond.size(0), cond.size(1), 1), device=cond.device)
        x0 = torch.randn_like(cond, device=cond.device) + cond
        xt = t * x1 + (1 - (1 - self.sigma_min) * t) * x0

        spk = self._apply_cfg_dropout(spk)
        vt = self.denoiser(xt, t.squeeze(-1), spk) * mask
        fm_loss = F.mse_loss(vt, (x1 - (1 - self.sigma_min) * x0) * mask)

        anchor_loss = F.mse_loss(cond * mask, x1 * mask)

        return {
            'fm_loss': fm_loss,
            'anchor_loss': anchor_loss,
        }

    def sample(
        self,
        cond,
        spk,
        mask,
        nfe=64,
        temperature=1.0,
        method='euler',
        prior_logits=None,
        guidance_scale=None,
        n_sampling_steps_min=None,
        n_sampling_steps_max=None,
    ):
        cond = self.quantizer_encoding(cond)
        cond = self.cond_downsampling(cond, mask)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        else:
            guidance_scale = float(guidance_scale)

        method = (method or 'euler').lower()
        use_cfg = self.cfg_dropout_prob > 0 and not math.isclose(guidance_scale, 1.0)
        spk_uncond = torch.zeros_like(spk) if use_cfg else None

        if method == 'forcing':
            return self._sample_forcing(
                cond,
                spk,
                mask,
                temperature=temperature,
                guidance_scale=guidance_scale,
                spk_uncond=spk_uncond,
                n_sampling_steps_min=n_sampling_steps_min,
                n_sampling_steps_max=n_sampling_steps_max,
                prior_logits=prior_logits,
            )
        if method == 'euler':
            return self._sample_euler(
                cond,
                spk,
                mask,
                nfe=nfe,
                temperature=temperature,
                guidance_scale=guidance_scale,
                spk_uncond=spk_uncond,
                prior_logits=prior_logits,
            )
        raise ValueError(f"Unsupported sampling method '{method}'")

    def _sample_euler(
            self, 
            cond, 
            spk, 
            mask, 
            nfe, 
            temperature=None, 
            guidance_scale=None, 
            spk_uncond=None,
            prior_logits=None
    ):
        if (temperature and prior_logits) or (not temperature and not prior_logits):
            raise ValueError('`temperature` and `prior_logits` are exclusive! Either `temperature` or `prior_logits` must be not None')
        
        b, l, _ = cond.shape
        ts = torch.linspace(0, 1, nfe + 1, device=cond.device)
        if prior_logits:
            temperature = self.mean_confidence(prior_logits)
            xt = torch.randn((b, l, self.target_dim), device=cond.device) * temperature + cond
        else:
            xt = torch.randn((b, l, self.target_dim), device=cond.device) * temperature + cond
        delta_t = 1 / nfe

        for i in range(1, len(ts)):
            vt = self._guided_velocity(
                xt,
                ts[i - 1].unsqueeze(0).unsqueeze(1),
                spk,
                spk_uncond=spk_uncond,
                mask=mask,
                guidance_scale=guidance_scale,
            )
            xt = xt + delta_t * vt

        return xt.transpose(1, -1)

    def _sample_forcing(
        self,
        cond,
        spk,
        mask,
        temperature=None,
        guidance_scale=None,
        spk_uncond=None,
        n_sampling_steps_min=None,
        n_sampling_steps_max=None,
        prior_logits=None,
    ):
        if n_sampling_steps_min is None or n_sampling_steps_max is None:
            raise ValueError("n_sampling_steps_min and n_sampling_steps_max must be provided for forcing sampling")

        n_sampling_steps_min = int(n_sampling_steps_min)
        n_sampling_steps_max = int(n_sampling_steps_max)

        if n_sampling_steps_min <= 0 or n_sampling_steps_max <= 0:
            raise ValueError("n_sampling_steps_min and n_sampling_steps_max must be positive integers")
        if n_sampling_steps_min > n_sampling_steps_max:
            raise ValueError("n_sampling_steps_min cannot be greater than n_sampling_steps_max")

        b, l, _ = cond.shape
        xt = torch.randn((b, l, self.target_dim), device=cond.device) * temperature + cond
        step_schedule = self._build_forcing_schedule(l, n_sampling_steps_min, n_sampling_steps_max, cond.device)
        total_steps = int(step_schedule.max().item())
        ts = torch.linspace(0, 1, total_steps + 1, device=cond.device)
        delta_t = 1 / total_steps

        for step_idx in range(1, total_steps + 1):
            step_mask = (step_idx <= step_schedule).view(1, l, 1)
            if mask is None:
                effective_mask = step_mask
            elif mask.dtype == torch.bool:
                effective_mask = mask & step_mask
            else:
                effective_mask = mask * step_mask.to(mask.dtype)

            vt = self._guided_velocity(
                xt,
                ts[step_idx - 1].unsqueeze(0).unsqueeze(1),
                spk,
                spk_uncond=spk_uncond,
                mask=effective_mask,
                guidance_scale=guidance_scale,
            )
            xt = xt + delta_t * vt

        return xt.transpose(1, -1)

    def _build_forcing_schedule(self, num_latents, steps_min, steps_max, device):
        if num_latents <= 0:
            raise ValueError("Number of latents must be positive")

        if num_latents == 1:
            return torch.tensor([max(steps_min, steps_max)], device=device, dtype=torch.long)

        schedule = torch.linspace(float(steps_min), float(steps_max), steps=num_latents, device=device)
        schedule = torch.round(schedule).to(torch.long)
        schedule[0] = steps_min
        schedule[-1] = steps_max

        for idx in range(1, num_latents):
            if schedule[idx] < schedule[idx - 1]:
                schedule[idx] = schedule[idx - 1]

        schedule = torch.clamp(schedule, min=1)
        return schedule

    def mean_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute mean confidence scores over quantizer dimension Q.

        Args:
            logits: Tensor shaped [B, C, Q, L] where C is class count.

        Returns:
            Tensor shaped [B, L] with the average softmax probability of the
            predicted (argmax) class across Q.
        """
        if logits.dim() != 4:
            raise ValueError(f"Expected logits with 4 dims [B, C, Q, L], got shape {tuple(logits.shape)}")

        probs = logits.softmax(dim=1)
        top_idx = logits.argmax(dim=1, keepdim=True)
        top_conf = probs.gather(1, top_idx).squeeze(1)
        return top_conf.mean(dim=2)
