import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim

        # Ensure window_size is a tuple of two integers
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size  # [window_height, window_width]

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define parameters
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads,
            )
        )  # [2*Wh-1 * 2*Ww-1, num_heads]

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, Wh, Ww]
        coords_flatten = coords.flatten(1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        x: [num_windows*B, N, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # [N, N, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: [B, H, W, C]
        window_size (int): window size
    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    
    # Check if H and W are divisible by window_size, if not pad the input
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    # Update H and W after padding
    H, W = x.shape[1], x.shape[2]
    
    # Reshape and permute to get the windows
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: [num_windows*B, window_size, window_size, C]
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: [B, H, W, C]
    """
    C = windows.shape[-1]  # Get the number of channels
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    
    # Check if the reshaping is possible
    if (H % window_size != 0) or (W % window_size != 0):
        raise ValueError(f"Height (H={H}) and Width (W={W}) must be divisible by the window size ({window_size}).")
    
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, C)
    )
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads

        # Ensure window_size is a tuple
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size

        self.mlp_ratio = mlp_ratio

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Window-based Multi-Head Self-Attention
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Calculate padding required to make H and W divisible by window size
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        
        # Apply padding if necessary
        x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad width first, then height
        
        # Update H and W after padding
        H_padded, W_padded = x.shape[2], x.shape[3]

        # Reshape and partition the padded feature map into windows
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_windows = window_partition(x, self.window_size[0])  # [num_windows*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # [num_windows*B, N, C]

        # Apply attention
        attn_windows = self.attn(x_windows)  # [num_windows*B, N, C]

        # Reshape back to windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        
        # Merge windows back into the image
        x = window_reverse(attn_windows, self.window_size[0], H_padded, W_padded)  # [B, H_padded, W_padded, C]

        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]  # Crop to original height and width
        
        # Permute back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()

        # MLP
        x = x.view(B, -1, C)  # [B, H*W, C]
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(x)

        x = x.view(B, C, H, W)  # Reshape back to [B, C, H, W]
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim, 2 * dim, bias=False
        )
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert (
            L == H * W
        ), "input feature has wrong size"
        assert (
            H % 2 == 0 and W % 2 == 0
        ), f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        x = torch.cat(
            [x0, x1, x2, x3], -1
        )  # [B, H/2, W/2, 4*C]
        x = x.view(
            B, -1, 4 * C
        )  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.0,
        downsample=None,
    ):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0
                    if (i % 2 == 0)
                    else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path,
                )
            )

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.0,
    ):
        super(SwinTransformer, self).__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path, sum(depths))
        ]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    img_size // (2 ** i_layer * patch_size),
                    img_size // (2 ** i_layer * patch_size),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(
                        depths[: i_layer + 1]
                    )
                ],
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(
                int(embed_dim * 2 ** (self.num_layers - 1)),
                num_classes,
            )
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, embed_dim]
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # [B, L, C]
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(
    x, drop_prob: float = 0.0, training: bool = False
):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (
        x.shape[0],
    ) + (1,) * (
        x.ndim - 1
    )  # Work with diff dimensions
    random_tensor = (
        keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    )
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
