import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        return self.dropout(torch.relu(self.ln(self.conv(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2))

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, num_action_types, num_conv_blocks=15):
        super().__init__()
        self.input_shape = input_shape
        linear = 512
        
        self.conv_blocks = nn.ModuleList([
            ConvBlock(input_shape[0]+2 if i == 0 else 64, 64)
            for i in range(num_conv_blocks)
        ])
        
        self.fc = nn.Flatten()
        self.shared_fc = nn.Linear(64 * input_shape[1] * input_shape[2] + 2, linear)
        self.shared_ln = nn.LayerNorm(linear)

        self.conv1_proj = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.output_grid_head = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.critic_fc = nn.Linear(linear, 1)

    def forward(self, grid, gold):
        gold_grid = gold.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid.size(2), grid.size(3))
        x = torch.cat((grid, gold_grid), dim=1)
        
        
        x = self.conv_blocks[0](x)
        projected_x1 = self.conv1_proj(x)
        
        
        for block in self.conv_blocks[1:]:
            x = block(x)

        shared_features = x

        x = self.fc(shared_features)
        x = torch.cat((x, gold), dim=1)
        x = torch.relu(self.shared_ln(self.shared_fc(x)))




        combined_input = shared_features + projected_x1

        output_grid = self.output_grid_head(combined_input)

        source_tile_logits = output_grid[:, 0, :, :]
        target_tile_logits = output_grid[:, 1, :, :]
        action_values = output_grid[:, 2, :, :]

        value = self.critic_fc(x)

        return action_values, source_tile_logits, target_tile_logits, value
    
    
class ResConvBlockConventional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(out_channels)
        

        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout2d(0.05)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_conv(x)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)
        
       
        
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = x + identity
        x = torch.relu(x)
        return self.dropout(x)    
    
    
class ResConvBlockFullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(out_channels)
        
        self.lin1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.ln_lin1 = nn.LayerNorm(out_channels)
        self.lin2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.ln_lin2 = nn.LayerNorm(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout2d(0.05)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_conv(x)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln1(x)
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.lin1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_lin1(x)
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.lin2(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_lin2(x)

        x = x.permute(0, 3, 1, 2)
        
        
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = x + identity
        x = torch.relu(x)
        return self.dropout(x)






class ResActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, num_action_types, block_type=ResConvBlockFullyConnected, num_res_blocks=3):
        super().__init__()
        self.input_shape = input_shape
        linear = 512
        
        self.res_blocks = nn.ModuleList([
            block_type(input_shape[0]+2 if i == 0 else 64, 64)
            for i in range(num_res_blocks)
        ])
        
        self.fc = nn.Flatten()
        self.shared_fc = nn.Linear(64 * input_shape[1] * input_shape[2] + 2, linear)
        self.shared_ln = nn.LayerNorm(linear)

        #self.conv1_proj = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.output_grid_head = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.critic_fc = nn.Linear(linear, 1)

    def forward(self, grid, gold):
        gold_grid = gold.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid.size(2), grid.size(3))
        x = torch.cat((grid, gold_grid), dim=1)
        
        x = self.res_blocks[0](x)
        #projected_x1 = self.conv1_proj(x)
        
        for block in self.res_blocks[1:]:
            x = block(x)

        shared_features = x
        
        x = self.fc(shared_features)
        x = torch.cat((x, gold), dim=1)
        x = torch.relu(self.shared_ln(self.shared_fc(x)))

        combined_input = shared_features #+ projected_x1

        output_grid = self.output_grid_head(combined_input)

        source_tile_logits = output_grid[:, 0, :, :]
        target_tile_logits = output_grid[:, 1, :, :]
        action_values = output_grid[:, 2, :, :]

        value = self.critic_fc(x)

        return action_values, source_tile_logits, target_tile_logits, value