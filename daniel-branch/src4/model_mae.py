import torch
import torch.nn as nn

class PatchEmbed1D(nn.Module):
    """把IQ序列切成patch，类似ViT"""
    def __init__(self, seq_len=768, patch_size=16, in_chans=2, embed_dim=128):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):  # x: [B, 2, 768]
        return self.proj(x).transpose(1, 2)  # [B, num_patches, embed_dim]


class IQMaskedAutoencoder(nn.Module):
    """
    RIS-MAE风格的MAE，适配你的IQ输入
    Pre-train: mask + reconstruct
    Fine-tune: 冻结encoder + L-SIG head
    """
    def __init__(
        self,
        seq_len=768,
        patch_size=16,       # 768/16 = 48 patches
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=4,
        mask_ratio=0.5,
        num_bits=24,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # Encoder
        self.patch_embed = PatchEmbed1D(seq_len, patch_size, 2, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder (only used during pre-training)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2)
        decoder_layer = nn.TransformerEncoderLayer(
            embed_dim // 2, num_heads, dim_feedforward=embed_dim*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(embed_dim // 2, patch_size * 2)  # reconstruct real+imag

        # Fine-tune head (L-SIG decoding)
        self.lsig_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_bits),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x):
        """随机mask掉mask_ratio比例的patches"""
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def encode(self, x, mask=True):
        """Encoder forward — mask=False时用于fine-tune"""
        tokens = self.patch_embed(x) + self.pos_embed
        if mask:
            tokens, mask_vec, ids_restore = self.random_masking(tokens)
            tokens = self.encoder(tokens)
            tokens = self.encoder_norm(tokens)
            return tokens, mask_vec, ids_restore
        else:
            tokens = self.encoder(tokens)
            tokens = self.encoder_norm(tokens)
            return tokens

    def forward_pretrain(self, x):
        """Pre-training forward: returns reconstruction loss"""
        tokens, mask_vec, ids_restore = self.encode(x, mask=True)

        # Decode
        tokens = self.decoder_embed(tokens)
        B, n_keep, D = tokens.shape
        n_masked = self.num_patches - n_keep
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        mask_tokens = self.decoder_embed(
            torch.zeros(B, n_masked, self.embed_dim, device=x.device)
        ) + mask_tokens[:, :, :D]

        # Restore original order
        full = torch.cat([tokens, mask_tokens], dim=1)
        full = torch.gather(
            full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        pred = self.decoder(full)
        pred = self.decoder_pred(pred)  # [B, N, patch_size*2]

        # Target: original patches
        target = self.patchify(x)
        loss = ((pred - target) ** 2)
        loss = (loss * mask_vec.unsqueeze(-1)).sum() / mask_vec.sum()
        return loss

    def patchify(self, x):
        """[B, 2, T] → [B, N, patch_size*2]"""
        B, C, T = x.shape
        p = self.patch_size
        x = x.reshape(B, C, T // p, p)
        x = x.permute(0, 2, 1, 3).reshape(B, T // p, C * p)
        return x

    def forward_finetune(self, x):
        """Fine-tune forward: L-SIG bit prediction"""
        tokens = self.encode(x, mask=False)     # [B, N, D]
        pooled = tokens.mean(dim=1)             # global average pool
        return self.lsig_head(pooled)           # [B, 24] logits