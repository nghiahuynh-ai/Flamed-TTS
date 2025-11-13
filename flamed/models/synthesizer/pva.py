import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from flamed.utils.tools import get_mask_from_lengths, pad


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim).float() * -emb)

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        emb = scale * x.unsqueeze(1) * self.emb.unsqueeze(0).to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        time_emb_scale,
        ):
        super(TimeEmbedding, self).__init__()
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * time_emb_scale),
            nn.SiLU(),
            nn.Linear(hidden_dim * time_emb_scale, hidden_dim)
        )
        
    def forward(self, t):
        return self.time_emb(t)


class PVA(nn.Module):
    """Probabilistic Variance Adaptor"""

    def __init__(self, model_config):
        super(PVA, self).__init__()
        self.sigma_min = model_config['sigma_min']
        self.duration_generator = ProbabilisticModule(model_config['duration_generator'])
        self.sil_generator = ProbabilisticModule(model_config['sil_generator'])
        self.length_regulator = LengthRegulator()

    def compute_loss(
        self,
        x,
        src_len,
        src_mask,
        max_tgt_len,
        phone_duration,
        sil_duration,
    ):
        t = torch.rand((x.shape[0], 1)).to(x.device)

        dur_1 = torch.log(phone_duration.float() + 1)
        dur_0 = torch.randn_like(dur_1)
        dur_t = t * dur_1 + (1 - (1 - self.sigma_min) * t) * dur_0
        u_dur = (dur_1 - (1 - self.sigma_min) * dur_0) * ~src_mask
        v_dur = self.duration_generator(dur_t, x, t.squeeze(), src_mask)
        dur_loss = F.mse_loss(v_dur, u_dur)
        
        sil_1 = torch.log(sil_duration.float() + 1)
        sil_0 = torch.randn_like(sil_1)
        sil_t = t * sil_1 + (1 - (1 - self.sigma_min) * t) * sil_0
        u_sil = (sil_1 - (1 - self.sigma_min) * sil_0) * ~src_mask
        v_sil = self.sil_generator(sil_t, x, t.squeeze(), src_mask)
        sil_loss = F.mse_loss(v_sil, u_sil)

        losses = {
            'dur_loss': dur_loss,
            'sil_loss': sil_loss,
        }

        x, _ = self.length_regulator(x, phone_duration, sil_duration, src_len, max_tgt_len)

        return x, losses
    
    def sample(
        self,
        x,
        src_len,
        src_mask,
        max_tgt_len=None,
        nfe=32,
        temperature=1.0,
    ):
        b, l, _ = x.size()
        ts = torch.linspace(0, 1, nfe + 1, device=x.device)
        delta_t = 1 / nfe

        dur_t = torch.randn((b, l)).to(x.device) * temperature
        sil_t = torch.randn((b, l)).to(x.device) * temperature
        
        for i in range(1, len(ts)):
            v_dur = self.duration_generator(dur_t, x, ts[i-1], src_mask)
            dur_t = dur_t + delta_t * v_dur

            v_sil = self.sil_generator(sil_t, x, ts[i-1], src_mask)
            sil_t = sil_t + delta_t * v_sil

        phone_duration = torch.clamp((torch.round(torch.exp(dur_t) - 1)), min=0)
        sil_duration = torch.clamp((torch.round(torch.exp(sil_t) - 1)), min=0)

        x, tgt_len = self.length_regulator(x, phone_duration, sil_duration, src_len, max_tgt_len)

        return x, tgt_len


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, phone_duration, sil_duration, src_lens, max_len):
        B, L, H = x.size()
        device = x.device

        phone_duration = phone_duration.to(device).float()
        sil_duration = sil_duration.to(device).float()
        src_lens = src_lens.to(device)

        idx = torch.arange(L, device=device).unsqueeze(0)
        valid_mask = idx < src_lens.unsqueeze(1)

        phone_repeat = torch.where(valid_mask, phone_duration, torch.zeros_like(phone_duration))
        phone_repeat = torch.clamp(phone_repeat.round().long(), min=1)

        sil_repeat = torch.where(valid_mask, sil_duration, torch.zeros_like(sil_duration))
        sil_repeat = torch.clamp(sil_repeat.round().long(), min=0)

        sil_frames = x[:, :1, :].expand(-1, L, -1).contiguous()

        segments = torch.stack((x, sil_frames), dim=2).reshape(B, -1, H)
        repeats = torch.stack((phone_repeat, sil_repeat), dim=2).reshape(B, -1)

        tgt_len = repeats.sum(dim=1)

        segments_flat = segments.reshape(-1, H)
        repeats_flat = repeats.reshape(-1)

        positive_mask = repeats_flat > 0
        if positive_mask.any():
            expanded = torch.repeat_interleave(segments_flat[positive_mask], repeats_flat[positive_mask], dim=0)
        else:
            expanded = segments_flat.new_zeros((0, H))

        split_sizes = tgt_len.tolist()
        output = list(torch.split(expanded, split_sizes, dim=0))

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, tgt_len.to(device)

    def forward(self, x, phone_duration, sil_duration, src_lens, max_len):
        output, tgt_len = self.LR(x, phone_duration, sil_duration, src_lens, max_len)
        return output, tgt_len
        

class ProbabilisticModule(nn.Module):
    """Probabilistic Module"""

    def __init__(self, model_config):
        super(ProbabilisticModule, self).__init__()

        self.input_size = model_config["input_size"]
        self.filter_size = model_config["filter_size"]
        self.kernel = model_config["kernel_size"]
        self.time_scale = model_config["time_scale"]
        self.conv_output_size = model_config["filter_size"]
        self.dropout = model_config["drop_out"]
        
        self.proj = nn.Linear(self.input_size + 1, self.input_size)
        self.time_emb = TimeEmbedding(self.input_size, self.time_scale)
        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, xt, encoder_output, t, mask):
        # Project
        out = self.proj(torch.cat([xt.unsqueeze(-1), encoder_output], dim=-1))
        
        # Time Embedding
        t = self.time_emb(t)
        t = t.unsqueeze(1).expand(-1, out.size(1), -1).contiguous()
        out = out + t

        # Vector Field Estimation
        out = self.conv_layer(out)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
