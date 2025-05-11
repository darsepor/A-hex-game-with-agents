import torch
import torch.nn as nn
import math
from res_net_AC import ResConvBlockOneLayerFullyConnected

VOCAB_SIZE = 9

class AttentionCNN(nn.Module):
    def __init__(self, embedding_dim, num_res_blocks=4, attention_heads=4, attention_dropout=0.1, num_attention_layers=2,
                 use_early_feature_skip=False, #The skip causes instability and makes the thing lazy it seems
                 block_type=ResConvBlockOneLayerFullyConnected):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_early_feature_skip = use_early_feature_skip
        self.tile_embedding = nn.Embedding(VOCAB_SIZE, self.embedding_dim)

        early_feature_dim = self.embedding_dim + 2

        cnn_channels = [early_feature_dim, 64, 128, 256]
        
        num_channel_values_needed = num_res_blocks + 1
        if num_channel_values_needed > len(cnn_channels):
            cnn_channels.extend([cnn_channels[-1]] * (num_channel_values_needed - len(cnn_channels)))

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_c = cnn_channels[i]
            out_c = cnn_channels[i+1]
            self.res_blocks.append(block_type(in_c, out_c))
        
        self.cnn_output_dim = cnn_channels[num_res_blocks]
        
        if self.use_early_feature_skip:
            raw_combined_dim = self.cnn_output_dim + early_feature_dim
        else:
            raw_combined_dim = self.cnn_output_dim
        
        self.norm_before_attn_proj = nn.LayerNorm(raw_combined_dim)

        if raw_combined_dim % attention_heads == 0:
            self.attention_d_model = raw_combined_dim
            self.pre_attention_projection = nn.Identity()
        else:
            self.attention_d_model = ((raw_combined_dim // attention_heads) + 1) * attention_heads
            self.pre_attention_projection = nn.Conv2d(raw_combined_dim, self.attention_d_model, kernel_size=1)

        self.pre_transformer_ln = nn.LayerNorm(self.attention_d_model)
        
        final_encoder_norm = nn.LayerNorm(self.attention_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_d_model, 
            nhead=attention_heads,
            dim_feedforward=self.attention_d_model * 4,
            dropout=attention_dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_attention_layers,
            norm=final_encoder_norm
        )
        
        self.pos_embed_cache = {}

        self.output_grid_head = nn.Sequential(
            nn.Conv2d(self.attention_d_model, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )
        
        self.critic_pool = nn.AdaptiveAvgPool2d((1,1))
        self.critic_flatten = nn.Flatten()
        self.critic_fc = nn.Sequential(
            nn.Linear(raw_combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _generate_2d_sincos_pos_embed(self, d_model, height, width, device):
        if self.pos_embed_cache.get('size') == (height, width) and \
           self.pos_embed_cache.get('dim') == d_model and \
           self.pos_embed_cache.get('device') == device:
            return self.pos_embed_cache['pos_embed']


        pos_embed = torch.zeros(1, d_model, height, width, device=device)
        half_d_model = d_model // 2
        
        div_term = torch.exp(torch.arange(0, half_d_model, 2, dtype=torch.float32, device=device) * \
                             -(math.log(10000.0) / half_d_model)) # Adjusted for half_d_model
        
        pos_w = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0)
        pos_h = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(0)

        pe_w_sin = torch.sin(pos_w.unsqueeze(-1) * div_term)
        pe_w_cos = torch.cos(pos_w.unsqueeze(-1) * div_term)
        pe_w = torch.cat([pe_w_sin, pe_w_cos], dim=-1).permute(0,2,1).unsqueeze(2)
        
        # For height
        pe_h_sin = torch.sin(pos_h.unsqueeze(-1) * div_term)
        pe_h_cos = torch.cos(pos_h.unsqueeze(-1) * div_term)
        pe_h = torch.cat([pe_h_sin, pe_h_cos], dim=-1).permute(0,2,1).unsqueeze(3)


        channels_w = pe_w.shape[1]
        channels_h = pe_h.shape[1]

        pos_embed[:, :channels_w, :, :] = pe_w.expand(-1, -1, height, -1)
        if d_model > channels_w : # If there's space left for height features
             pos_embed[:, channels_w:channels_w+channels_h, :, :] = pe_h.expand(-1, -1, -1, width)


        self.pos_embed_cache = {'pos_embed': pos_embed, 'size': (height, width), 'dim': d_model, 'device': device}
        return pos_embed

    def forward(self, grid, gold):
        B, H, W = grid.shape
        device = grid.device
        
        embedded_tiles = self.tile_embedding(grid.long())
        embedded_tiles = embedded_tiles.permute(0, 3, 1, 2)

        gold_expanded = gold.unsqueeze(-1).unsqueeze(-1).expand(B, gold.size(1), H, W)
        
        early_features = torch.cat((embedded_tiles, gold_expanded), dim=1)
        
        cnn_processed_features = early_features
        for i, block in enumerate(self.res_blocks):
            cnn_processed_features = block(cnn_processed_features) 
        
        deep_cnn_features = cnn_processed_features

        if self.use_early_feature_skip:
            combined_features_for_proj = torch.cat((deep_cnn_features, early_features), dim=1)
        else:
            combined_features_for_proj = deep_cnn_features


        critic_pooled_cnn_features = self.critic_pool(combined_features_for_proj)
        critic_flat_cnn_features = self.critic_flatten(critic_pooled_cnn_features)
        value = self.critic_fc(critic_flat_cnn_features) # This 'value' is used in the return

        norm_input_for_proj = combined_features_for_proj.permute(0, 2, 3, 1) 
        norm_output_for_proj = self.norm_before_attn_proj(norm_input_for_proj)
        projected_input = norm_output_for_proj.permute(0, 3, 1, 2) 
        
        projected_for_attention = self.pre_attention_projection(projected_input)

        current_attention_d_model = projected_for_attention.shape[1] 
        pos_encoding = self._generate_2d_sincos_pos_embed(current_attention_d_model, H, W, device)
            
        features_with_pos = projected_for_attention + pos_encoding
            
        seq_for_transformer = features_with_pos.flatten(2).permute(2, 0, 1)
        
        seq_for_transformer = self.pre_transformer_ln(seq_for_transformer)
            
        attended_seq = self.transformer_encoder(seq_for_transformer)
            
        attended_features_grid = attended_seq.permute(1, 2, 0).view(B, current_attention_d_model, H, W)

        output_grid_logits = self.output_grid_head(attended_features_grid)
        
        source_tile_logits = output_grid_logits[:, 0, :, :]
        target_tile_logits = output_grid_logits[:, 1, :, :]
        action_values = output_grid_logits[:, 2, :, :]

        return action_values, source_tile_logits, target_tile_logits, value.squeeze(-1)

