import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE = 9  # EMPTY_LAND, EMPTY_WATER, P1_SOLDIER, ..., OUT_OF_BOUNDS
HEX_NEIGHBOR_OFFSETS = [
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1) # q, r offsets for neighbors radius 1
]
NUM_NEIGHBORS_INC_SELF = len(HEX_NEIGHBOR_OFFSETS)

class HexSlidingAttentionLayer(nn.Module):
    """
    Performs multi-head self-attention where each token (tile) attends
    only to itself and its 6 direct hexagonal neighbors.
    Boundary handling uses bounds checking; invalid neighbors get zero vectors.
    """
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def _get_hex_neighbors(self, x, H, W):
        """
        Gathers features from hexagonal neighbors for each tile using bounds checking.
        Returns gathered features where invalid neighbors are represented by zero vectors.

        Args:
            x (Tensor): Input tensor (B, H, W, C)
            H (int): Height
            W (int): Width

        Returns:
            Tensor: gathered_features (B, H, W, NUM_NEIGHBORS_INC_SELF, C)
        """
        B, H_in, W_in, C = x.shape
        assert H_in == H and W_in == W

        device = x.device

        gathered_features = torch.zeros(B, H, W, NUM_NEIGHBORS_INC_SELF, C, device=device)

        grid_r, grid_q = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

        batch_idx = torch.arange(B, device=device)[:, None, None] # Shape (B, 1, 1)

        for i, (dq, dr) in enumerate(HEX_NEIGHBOR_OFFSETS):
            neighbor_q = grid_q + dq
            neighbor_r = grid_r + dr

            valid_mask = (neighbor_q >= 0) & (neighbor_q < W) & \
                         (neighbor_r >= 0) & (neighbor_r < H)
            # valid_mask shape: (H, W)

            selected_q = torch.where(valid_mask, neighbor_q, 0)
            selected_r = torch.where(valid_mask, neighbor_r, 0)

            features_at_neighbors = x[batch_idx, selected_r, selected_q, :] # Shape (B, H, W, C)

            # Unsqueeze mask to (1, H, W, 1) for broadcasting with (B, H, W, C)
            valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(-1) # Shape (1, H, W, 1)
            gathered_features[:, :, :, i, :] = features_at_neighbors * valid_mask_expanded

        return gathered_features


    def forward(self, src):
        """
        Input shape: (B, H, W, C)
        """
        B, H, W, C = src.shape
        x = src

        # gathered_k/v: (B, H, W, 7, C)
        gathered_kv = self._get_hex_neighbors(x, H, W)

        # Reshape x (query): (B*H*W, 1, C)
        # Reshape gathered_kv (key/value): (B*H*W, 7, C)
        q = x.reshape(B*H*W, 1, C)
        kv = gathered_kv.reshape(B*H*W, NUM_NEIGHBORS_INC_SELF, C)

        # Query is the center tile, Key/Value are the neighbors (including self, with zeros for invalid)
        attn_output, attn_weights = self.self_attn(q, kv, kv, attn_mask=None, need_weights=False)
        # attn_output shape: (B*H*W, 1, C)

        # Reshape output back to grid: (B, H, W, C)
        attn_output = attn_output.reshape(B, H, W, C)

        x = x + self.dropout1(attn_output) # Residual connection
        x = self.norm1(x)

        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output) # Residual connection
        x = self.norm2(x)

        return x # Shape: (B, H, W, C)


class HexTransformer(nn.Module):
    def __init__(self, embedding_dim=22, num_layers=8, attention_heads=4, attention_dropout=0.1,
                 mlp_hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.tile_embedding = nn.Embedding(VOCAB_SIZE, self.embedding_dim)
        self.input_feature_dim = self.embedding_dim + 2
        self.d_model = self.input_feature_dim
        if self.d_model % attention_heads != 0:
             raise NotImplementedError("d_model must be divisible by attention_heads, or add a projection layer.")

        self.pos_embed_cache = {}

        self.layers = nn.ModuleList([
            HexSlidingAttentionLayer(
                d_model=self.d_model,
                nhead=attention_heads,
                dropout=attention_dropout
            ) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(self.d_model)

        self.policy_mlp = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 3) # 3 outputs: src_logit, tgt_logit, action_value
        )

        self.critic_pool = nn.AdaptiveAvgPool2d((1,1))
        self.critic_flatten = nn.Flatten()
        self.critic_fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Single value output
        )

    def _generate_2d_sincos_pos_embed(self, d_model, height, width, device):
        if self.pos_embed_cache.get('size') == (height, width) and \
           self.pos_embed_cache.get('dim') == d_model and \
           self.pos_embed_cache.get('device') == device:
            return self.pos_embed_cache['pos_embed']

        pos_embed = torch.zeros(1, d_model, height, width, device=device) # B=1, C, H, W
        half_d_model = d_model // 2

        div_term = torch.exp(torch.arange(0, half_d_model, 2, dtype=torch.float32, device=device) * \
                             -(math.log(10000.0) / half_d_model))

        pos_w = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0) # (1, W)
        pos_h = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(0) # (1, H)

        pe_w_sin = torch.sin(pos_w.unsqueeze(-1) * div_term) # (1, W, channels/4)
        pe_w_cos = torch.cos(pos_w.unsqueeze(-1) * div_term) # (1, W, channels/4)
        pe_w = torch.cat([pe_w_sin, pe_w_cos], dim=-1) # (1, W, channels/2)
        pe_w = pe_w.permute(0, 2, 1).unsqueeze(2) # (1, channels/2, 1, W)

        pe_h_sin = torch.sin(pos_h.unsqueeze(-1) * div_term) # (1, H, channels/4)
        pe_h_cos = torch.cos(pos_h.unsqueeze(-1) * div_term) # (1, H, channels/4)
        pe_h = torch.cat([pe_h_sin, pe_h_cos], dim=-1) # (1, H, channels/2)
        pe_h = pe_h.permute(0, 2, 1).unsqueeze(3) # (1, channels/2, H, 1)

        channels_w = pe_w.shape[1]
        channels_h = pe_h.shape[1]
        pos_embed[:, :channels_w, :, :] = pe_w.expand(-1, -1, height, -1)
        if d_model > channels_w :
            pos_embed[:, channels_w:channels_w+channels_h, :, :] = pe_h.expand(-1, -1, -1, width)

        self.pos_embed_cache = {'pos_embed': pos_embed, 'size': (height, width), 'dim': d_model, 'device': device}
        # Returns shape (1, C, H, W)
        return pos_embed

    def forward(self, grid, gold):
        """
        Args:
            grid (Tensor): Input grid (B, H, W) of tile IDs
            gold (Tensor): Gold values (B, 2) for current player and opponent

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - action_values (Tensor): (B, H, W)
                - source_tile_logits (Tensor): (B, H, W)
                - target_tile_logits (Tensor): (B, H, W)
                - value (Tensor): (B,) state value estimate
        """
        B, H, W = grid.shape
        device = grid.device

        embedded_tiles = self.tile_embedding(grid.long()) # (B, H, W, embed_dim)
        gold_expanded = gold.unsqueeze(1).unsqueeze(1).expand(B, H, W, gold.size(1)) # (B, H, W, 2)
        features = torch.cat((embedded_tiles, gold_expanded), dim=-1) # (B, H, W, embed_dim + 2)
        # Note: Features are (B, H, W, C)

        pos_encoding = self._generate_2d_sincos_pos_embed(self.d_model, H, W, device) # (1, C, H, W)
        pos_encoding = pos_encoding.permute(0, 2, 3, 1) # (1, H, W, C)
        features = features + pos_encoding

        x = features
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x) # Final normalization -> (B, H, W, C)

        # Needs (B, C, H, W) -> (B, C, 1, 1)
        critic_pooled = self.critic_pool(x.permute(0, 3, 1, 2))
        critic_flat = self.critic_flatten(critic_pooled) # (B, C)
        value = self.critic_fc(critic_flat).squeeze(-1) # (B,)

        policy_mlp_input = x # (B, H, W, C)
        policy_mlp_input_flat = policy_mlp_input.view(B * H * W, self.d_model)
        policy_output_flat = self.policy_mlp(policy_mlp_input_flat) # (B*H*W, 3)
        policy_output_grid = policy_output_flat.view(B, H, W, 3) # (B, H, W, 3)

        source_tile_logits = policy_output_grid[:, :, :, 0] # (B, H, W)
        target_tile_logits = policy_output_grid[:, :, :, 1] # (B, H, W)
        action_values = policy_output_grid[:, :, :, 2]      # (B, H, W)

        return action_values, source_tile_logits, target_tile_logits, value 