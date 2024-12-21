import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from hex_game_env import CustomGameEnv
import sys
import os
from collections import deque
import time
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter





def sample_apply_masks(action_type_logits, source_tile_logits, target_tile_logits, state, env):
    # Create tensors from state
    unit_tensor = torch.tensor(state["grid"][1]).float().unsqueeze(0).cuda()
    land_water_tensor = torch.tensor(state["grid"][0]).float().unsqueeze(0).cuda()
    terrain_mask = env.mask.cuda()
    
    # Sample action type
    action_type_probs = torch.softmax(action_type_logits, dim=-1)
    action_type_distribution = torch.distributions.Categorical(action_type_probs)
    action_type = action_type_distribution.sample()
    
    # Force action type based on conditions
    if torch.all((unit_tensor>7) | (unit_tensor<=0)):
        action_type = torch.tensor(1).cuda()
    if torch.tensor(state["gold"][0]).float()<=0:
        action_type = torch.tensor(0).cuda()
    
    # Source tile masking
    source_mask = torch.where(unit_tensor <= 0, torch.tensor(float('-inf')), torch.tensor(0.0)).cuda()
    if(action_type==0):
        source_mask = source_mask + torch.where(unit_tensor == 20, torch.tensor(float('-inf')), torch.tensor(0.0))
    
    masked_source_logits = torch.clamp(source_tile_logits + source_mask, min=-1e9)
    source_tile_probs = torch.softmax((masked_source_logits).view(-1), dim=-1)
    source_tile_distribution = torch.distributions.Categorical(source_tile_probs)
    source_tile_idx = source_tile_distribution.sample()
    
    # Convert source tile index to coordinates
    source_tile_idx_int = source_tile_idx.item()
    source_tile_q = source_tile_idx_int // env.size - env.game.size
    source_tile_r = source_tile_idx_int % env.size - env.game.size
    
    # Calculate masking coordinates
    q_masking = source_tile_idx_int // env.size
    r_masking = source_tile_idx_int % env.size
    
    # Create distance mask
    q_coords, r_coords = torch.meshgrid(torch.arange(env.size), torch.arange(env.size), indexing='ij')
    dist = (torch.abs(q_coords - q_masking) + torch.abs(r_coords - r_masking) + 
           torch.abs((-r_coords - q_coords) - (-r_masking - q_masking))) // 2
    dist = dist.cuda()
    
    # Apply different conditions based on action type and unit type
    if action_type == 1:
        condition = (dist == 1)
    elif unit_tensor.squeeze(0)[q_masking][r_masking].item()==3:
        condition = (dist == 1) & (((unit_tensor == 0.0) & (land_water_tensor==1)) | (unit_tensor<0))
    else:
        condition = (dist>0) & (dist<=2) & (((unit_tensor == 0) & (land_water_tensor==0)) | (unit_tensor<0))
    
    target_mask = torch.where(condition, torch.tensor(0.0), torch.tensor(float('-inf'))).cuda()
    filled_mask = torch.where(unit_tensor > 0, torch.tensor(float('-inf')), torch.tensor(0.0))
    masked_target_logits = torch.clamp(target_tile_logits + target_mask + terrain_mask + filled_mask, min=-1e9)
    
    target_tile_probs = torch.softmax((masked_target_logits).view(-1), dim=-1)
    target_tile_distribution = torch.distributions.Categorical(target_tile_probs)
    target_tile_idx = target_tile_distribution.sample()
    
    target_tile_idx_int = target_tile_idx.item()
    target_tile_q = target_tile_idx_int // env.size - env.game.size
    target_tile_r = target_tile_idx_int % env.size - env.game.size
    
    return {
        'action_type': action_type,
        'action_type_distribution': action_type_distribution,
        'source_tile_idx': source_tile_idx,
        'source_tile_distribution': source_tile_distribution,
        'target_tile_idx': target_tile_idx,
        'target_tile_distribution': target_tile_distribution,
        'coordinates': {
            'source_q': source_tile_q,
            'source_r': source_tile_r,
            'target_q': target_tile_q,
            'target_r': target_tile_r
        }
    }

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
            ConvBlock(num_action_types if i == 0 else 64, 64)
            for i in range(num_conv_blocks)
        ])
        
        self.fc = nn.Flatten()
        self.shared_fc = nn.Linear(64 * input_shape[1] * input_shape[2] + 2, linear)
        self.shared_ln = nn.LayerNorm(linear)
        self.action_type_head = nn.Linear(linear, num_action_types)

        self.action_info_proj = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=1)
        self.conv1_proj = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.output_grid_head = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.critic_fc = nn.Linear(linear, 1)

    def forward(self, grid, gold):
        x = grid
        for block in self.conv_blocks:
            x = block(x)

        shared_features = x

        x = self.fc(shared_features)
        x = torch.cat((x, gold), dim=1)
        x = torch.relu(self.shared_ln(self.shared_fc(x)))

        action_type_logits = self.action_type_head(x)

        action_info = action_type_logits.unsqueeze(-1).unsqueeze(-1)
        action_info = action_info.expand(-1, 2, self.input_shape[1], self.input_shape[2])

        projected_action_info = self.action_info_proj(action_info)
        projected_x1 = self.conv1_proj(self.conv_blocks[0](grid))

        combined_input = shared_features + projected_action_info + projected_x1

        output_grid = self.output_grid_head(combined_input)

        source_tile_logits = output_grid[:, 0, :, :]
        target_tile_logits = output_grid[:, 1, :, :]

        value = self.critic_fc(x)

        return action_type_logits, source_tile_logits, target_tile_logits, value

def main():
    epochs = 100
    learning_rate = 0.001
    discount_factor = 0.99 #gamma
    trace_decay_rate = 0.9 #lambda
    #initialized = False #whether we're not training from scratch
    size = 5

    env = CustomGameEnv(size)
    
    
    model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()
    target_model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()
    
    #saved_state_dict = torch.load(f"size-{size}-actor_critic_model.pth")
    #model.load_state_dict(saved_state_dict)
    
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    total_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    critic_loss_fn = nn.MSELoss()

    victories = 0
    cumulative_rewards = deque(maxlen=10)
    cumulative_rewards.append(0)
    rewards_per_timestep = []

    for epoch in range(epochs):
        episode_reward = 0
        eligibility_traces_player1 = {name: torch.zeros_like(param, device='cuda') for name, param in model.named_parameters()}
        eligibility_traces_player2 = {name: torch.zeros_like(param, device='cuda') for name, param in model.named_parameters()}
        
        state = env.reset(size)
        done = False
        timestep = -1
        print(f"Epoch: {epoch}")
        
        invalid = 0
        
        
        while not done:
            timestep += 1
            grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).cuda()
            gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).cuda()
            
            action_type_logits, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
            
            sampling_results = sample_apply_masks(action_type_logits, source_tile_logits, target_tile_logits, state, env)
            
            action_type = sampling_results['action_type']
            source_tile_idx = sampling_results['source_tile_idx']
            target_tile_idx = sampling_results['target_tile_idx']
            coords = sampling_results['coordinates']
            
            player = env.game.current_player_index
            
            next_state_this_pov, next_state_next_pov, reward, done, _ = env.step(                               #No good!
                (action_type, coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r']))
            
            episode_reward += reward
            

            
            if reward < 0:
                invalid += 1
            
            next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).float().unsqueeze(0).cuda()
            next_gold_tensor = torch.tensor(next_state_this_pov["gold"]).float().unsqueeze(0).cuda()
            
            with torch.no_grad():
                _, _, _, next_value = target_model(next_grid_tensor, next_gold_tensor)
                td_target = reward + discount_factor * next_value * (1 - int(done))
            
            critic_loss = critic_loss_fn(value, td_target)
            advantage = (td_target - value).detach()
            
            log_prob_action_type = sampling_results['action_type_distribution'].log_prob(action_type)
            log_prob_source_tile = sampling_results['source_tile_distribution'].log_prob(source_tile_idx)
            log_prob_target_tile = sampling_results['target_tile_distribution'].log_prob(target_tile_idx)
            
            actor_loss = -(log_prob_action_type + log_prob_source_tile + log_prob_target_tile) 
            
            total_loss = actor_loss * advantage + critic_loss
            
            total_optimizer.zero_grad()
            total_loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if timestep % 2 == 0:
                        eligibility_traces_player1[name] = (trace_decay_rate * eligibility_traces_player1[name] + param.grad)
                        param.grad = eligibility_traces_player1[name]
                    else:
                        eligibility_traces_player2[name] = (trace_decay_rate * eligibility_traces_player2[name] + param.grad)
                        param.grad = eligibility_traces_player2[name] * advantage
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            

            
            total_optimizer.step()
            
            tau = 0.01
            for target_param, local_param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                
            
            if done or (reward > 0 and epoch < 100) or timestep % 100 == 0:
                grid = state["grid"][1]
                grid_str = '\n'.join([' '.join(map(str, row)) for row in grid])
                os.system('cls')
                sys.stdout.write(str(state["gold"]) + "\n")
                sys.stdout.flush()
                sys.stdout.write(grid_str)
                sys.stdout.flush()
                sys.stdout.write(f"\n Step: {timestep}, player: {player}, reward: {reward}, Epoch: {epoch}, " + 
                               f"action: {(action_type.item(), coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r'])}\n" +
                               f"victories: {victories}, value, next_value: {value.item(), next_value.item()} \n" +
                               f"invalid: {invalid}, avg_reward: {sum(cumulative_rewards) / len(cumulative_rewards)}")
                sys.stdout.flush()
            
            state = next_state_next_pov
            
            if done:
                victories += 1
            elif timestep > 5000:
                done = True
        
        cumulative_rewards.append(episode_reward/timestep)
        rewards_per_timestep.append(episode_reward/timestep)

    torch.save(model.state_dict(), f"size-{size}-actor_critic_model.pth")
    print("Training complete!!")

    rewards_file = "rewards.json"
    
    if os.path.exists(rewards_file):
        with open(rewards_file, "r") as f:
            all_rewards = json.load(f)
    else:
        all_rewards = []


    all_rewards.extend(rewards_per_timestep)

    with open(rewards_file, "w") as f:
        json.dump(all_rewards, f)

    plt.figure(figsize=(8, 5))
    plt.plot(list(range(len(all_rewards))), all_rewards, label="Average Reward per episode")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per timestep")
    plt.title("Aggregated Rewards Over Multiple Runs")
    plt.legend()
    plt.grid(True)
    plt.savefig("./plots/aggregated_rewards.png")
    plt.show()
    
    
    
if __name__ == "__main__":
    main()