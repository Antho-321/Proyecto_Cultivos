import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    Divide el tensor x en ventanas de tamaño window_size x window_size.
    Input:
        x: tensor de forma (B, H, W, C)
    Return:
        windows: tensor de forma (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    # permutar y aplanar
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
               .view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reconstruye el tensor original a partir de las ventanas.
    Input:
        windows: tensor (num_windows*B, window_size, window_size, C)
        H, W: dimensiones originales
    Return:
        x: tensor de forma (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
         .view(B, H, W, -1)
    return x

class ShiftedWindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        """
        dim: número de canales de entrada
        window_size: tamaño de la ventana (int)
        num_heads: número de cabezas de atención
        shift_size: tamaño del desplazamiento (0 o window_size//2)
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Sesgo de posición relativo
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1),
                        num_heads)
        )  # (M*M, num_heads), M = 2*W-1

        # Índice de posición relativa precomputado
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))  # (2, W, W)
        coords_flatten = coords.flatten(1)  # (2, W*W)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, W*W, W*W)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (W*W, W*W, 2)
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (W*W, W*W)
        self.register_buffer("relative_position_index", relative_position_index)

        # Proyecciones qkv y salida
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        """
        x: tensor de forma (B, H, W, C)
        """
        B, H, W, C = x.shape
        ws = self.window_size
        ss = self.shift_size

        # aplicar padding si H o W no son múltiplos de window_size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # desplazamiento cíclico
        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))

        # particionar en ventanas
        x_windows = window_partition(x, ws)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, ws * ws, C)  # (nW*B, N, C), N = ws*ws

        # calcular q, k, v
        qkv = self.qkv(x_windows) \
                   .reshape(-1, ws * ws, 3, self.num_heads, C // self.num_heads) \
                   .permute(2, 0, 3, 1, 4)  # (3, nW*B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # atención
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (nW*B, heads, N, N)
        # añadir sesgo de posición relativo
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1)  # (N, N, heads)
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + relative_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # salida de la atención
        x_attn = (attn @ v)  # (nW*B, heads, N, head_dim)
        x_attn = x_attn.transpose(1, 2).reshape(-1, ws * ws, C)  # (nW*B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # restaurar forma espacial
        x_attn = x_attn.view(-1, ws, ws, C)
        x = window_reverse(x_attn, ws, Hp, Wp)  # (B, Hp, Wp, C)

        # invertir desplazamiento cíclico
        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))

        # recortar padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        return x

# Ejemplo de uso dentro de un bloque de Swin Transformer
# x: tensor de forma (B, H, W, C)
# msa = ShiftedWindowAttention(
#     dim=C,
#     window_size=7,
#     num_heads=8,
#     shift_size=3,      # normalmente window_size // 2
#     qkv_bias=True,
#     attn_drop=0.1,
#     proj_drop=0.1
# )
# out = msa(x)  # (B, H, W, C)