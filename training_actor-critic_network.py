import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from hex_game_env import CustomGameEnv
import sys
import os
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, num_action_types):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Flatten()
        
        self.shared_fc = nn.Linear(64 * input_shape[1] * input_shape[2] + 2, 256)  #Shared layer after convolution
        #print(self.shared_fc)
        self.action_type_head = nn.Linear(256, num_action_types)
        self.source_tile_head = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.target_tile_head = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1) #we dont give a pathing scaffolding,
                                                                                             #nor do we mask invalid actions,
                                                                                             # so logits much be close
        
        self.critic_fc = nn.Linear(256, 1)

    def forward(self, grid, gold):
        x = torch.relu(self.conv1(grid))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        shared_features = x
        x = self.fc(x)
        
        x = torch.cat((x, gold), dim=1)
        x = torch.relu(self.shared_fc(x))
        
        action_type_logits = self.action_type_head(x)
        
        source_tile_logits = (self.source_tile_head(shared_features)).squeeze(1)
        target_tile_logits = (self.target_tile_head(source_tile_logits)).squeeze(1)
        
        value = self.critic_fc(x)

        return action_type_logits, source_tile_logits, target_tile_logits, value #further masking of target logits below

epochs = 100
learning_rate = 0.001
discount_factor = 0.9
size=8
env = CustomGameEnv(size)
#print(env.size)
model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()

total_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

critic_loss_fn = nn.MSELoss()

for epoch in range(epochs):
    state = env.reset()
    done = False
    total_reward = 0
    timestep = -1
    print(f"Epoch: {epoch}")
    while not done:
        timestep +=1
        grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).cuda()
        gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).cuda()
        unit_tensor = torch.tensor(state["grid"][1]).float().unsqueeze(0).cuda()
        
        terrain_mask = env.mask.cuda()
        
        source_mask = torch.where(unit_tensor <= 0, torch.tensor(float('-inf')), unit_tensor)
        
        action_type_logits, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
        
        action_type_probs = torch.softmax(action_type_logits, dim=-1)
        action_type_distribution = torch.distributions.Categorical(action_type_probs)
        action_type = action_type_distribution.sample()
        
        source_tile_probs = torch.softmax((source_tile_logits + source_mask).view(-1), dim=-1)
        source_tile_distribution = torch.distributions.Categorical(source_tile_probs)
        source_tile_idx = source_tile_distribution.sample()
        source_tile_idx_int = source_tile_idx.item()
        source_tile_q = source_tile_idx_int // env.size - env.game.size
        source_tile_r = source_tile_idx_int % env.size - env.game.size
        
        q_masking = source_tile_idx_int // env.size
        r_masking = source_tile_idx_int % env.size
        
        s_masking = -q_masking - r_masking
        
        q_coords, r_coords = torch.meshgrid(torch.arange(env.size), torch.arange(env.size), indexing='ij')
        dist = (torch.abs(q_coords - q_masking) + torch.abs(r_coords - r_masking) + torch.abs((-r_coords - q_coords) - (-r_masking - q_masking))) / 2
        target_mask = torch.where(dist <= 2, torch.tensor(0.0), torch.tensor(float('-inf'))).cuda() + terrain_mask
        
        
        
        target_tile_probs = torch.softmax((target_tile_logits + target_mask).view(-1), dim=-1)
        target_tile_distribution = torch.distributions.Categorical(target_tile_probs)
        target_tile_idx = target_tile_distribution.sample()
        target_tile_idx_int = target_tile_idx.item()
        target_tile_q = target_tile_idx_int // env.size - env.game.size
        target_tile_r = target_tile_idx_int % env.size - env.game.size
        player = env.game.current_player_index
        next_state, reward, done, _ = env.step((action_type, source_tile_q, source_tile_r, target_tile_q, target_tile_r))
        total_reward += reward
        
        next_grid_tensor = torch.tensor(next_state["grid"]).float().unsqueeze(0).cuda()
        next_gold_tensor = torch.tensor(next_state["gold"]).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, _, _, next_value = model(next_grid_tensor, next_gold_tensor)
            td_target = reward + discount_factor * next_value * (1 - int(done))
        
        critic_loss = critic_loss_fn(value, td_target)
        
        advantage = (td_target - value).detach()
        
        log_prob_action_type = action_type_distribution.log_prob(action_type)
        log_prob_source_tile = source_tile_distribution.log_prob(source_tile_idx)
        log_prob_target_tile = target_tile_distribution.log_prob(target_tile_idx)
        actor_loss = -(log_prob_action_type + log_prob_source_tile + log_prob_target_tile) * advantage
        
        total_loss = actor_loss + critic_loss
        
        total_optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_optimizer.step()
        if reward > 1 or timestep % 1000 == 0: #more frequent rewards now that a mask was added
            grid = state["grid"][1]
            grid_str = '\n'.join([' '.join(map(str, row)) for row in grid])
            new = next_state["grid"][1]
            new_str = '\n'.join([' '.join(map(str, row)) for row in new])
            os.system('cls')
            #print(source_mask)
            #print(env.game.atlas.get_hex(source_tile_q, source_tile_r, -source_tile_q - source_tile_r).unit is not None)
            sys.stdout.write(grid_str)
            sys.stdout.flush()
            sys.stdout.write(f"\n Step: {timestep}, player: {player}, reward: {reward}, Epoch: {epoch}, action: {(action_type.item(), source_tile_q, source_tile_r, target_tile_q, target_tile_r)}\n",)
            sys.stdout.flush()
            #sys.stdout.write(new_str)
            #sys.stdout.flush()
        state = next_state
        

    if epoch % 10 == 0:
        print(f"Ep Reward: {total_reward}")

torch.save(model.state_dict(), 'actor_critic_model.pth')

print("Training complete!!")
