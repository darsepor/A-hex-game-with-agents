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

'''
#Thought process doc/reminders:
 
we dont give a pathing scaffolding, nor do we mask invalid actions, so logits much be close
update: had to use maybe too many masks to barely work
(feels like "might as well write a bunch of if statements"), better approach needed

update2: maybe an even more extensive/adaptive mask and rewarding only
winning and nothing else would still produce something interesting 
(ie, focusing on strategy instead of picking out "valid" actions),
intermediate rewards depending on how many pieces player has could help with the 'sparsity'
also, could maybe train against (or to imitate) an improved SimpleAI
and punish for losing against it

update3: made game more strategic instead of a stalemate slog by adding an economy,
masking being overhauled is planned next as even now most actions are invalid. 
Also, fixed a massive oversight of the next_state the
network sees being from the POV of the opponent.
I'm using the proportion of how many games get resolved within a 1000 timesteps 
as a preliminary evaluation heurisic
at convergence and the best performance so far is around 30%. Getting better!

update4: 40-50%
                                                                                             
19/12

Found the time and motivation to come back to this.
Okay, turns out increasing lambda is all you need. Which is obvious since
we have a strategy game here.

20/12

-I've decided to use reward per timestep as an evaluation metric as it is higher
if the model gets on with it quickly. I suspect there are soft-lock states 
that mess with training. 
-Still haven't hardcoded complete masking.
-I think shorter episodes are preferred as the states become less correlated and
the end game states aren't that different from beginning ones anyway. There're states where the
AI is stuck though and does many invalid actions and 
maybe it's good for the value head to discourage them... No clue.
-The soft-lock states may be that the AI just spent all their money and there's no path to the enemy
due to terrain. (Say, it built only soldiers but there's a body of water that needs to be crossed).
Maybe changing the game logic to include cash-outs would work but that would mess with debt mechanic and unit
costs. And it seems I've tuned those quite well two months ago because touching them just seems to
make the game dumber/harder to resolve, or at least seems so from watching the visualization during training.
-TO DO: Should include terrain in the terminal visualization. Should track invalid actions.
'''

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
            ConvBlock(num_action_types if i == 0 else 64,
                      64)
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





epochs = 50
learning_rate = 0.001
discount_factor = 0.99 #gamma
trace_decay_rate = 0.9#lambda
initialized = False #whether we're not training from scratch

size = 5

env = CustomGameEnv(size)

model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()
target_model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()


saved_state_dict = torch.load("size-5-actor_critic_model.pth")
model.load_state_dict(saved_state_dict)


target_model.load_state_dict(model.state_dict())
target_model.eval()

#initialized = True


total_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.01)

critic_loss_fn = nn.MSELoss()

victories = 0
cumulative_rewards = deque(maxlen = 10)
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
    recent_actions = deque(maxlen=500)
    
    last_action = 0
    repeats = 0
    I = 1
    invalid = 0
    target_model.load_state_dict(model.state_dict())
    while not done:
        timestep +=1
        grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).cuda()
        gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).cuda()
        unit_tensor = torch.tensor(state["grid"][1]).float().unsqueeze(0).cuda()
        land_water_tensor = torch.tensor(state["grid"][0]).float().unsqueeze(0).cuda()
        

        
        action_type_logits, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
       # temperature = 2.0
       
        action_type_logits = action_type_logits
        action_type_probs = torch.softmax(action_type_logits, dim=-1)
        action_type_distribution = torch.distributions.Categorical(action_type_probs)
        action_type = action_type_distribution.sample()
        
        
        
        
        
        
        terrain_mask = env.mask.cuda()
        
        source_mask = torch.where(unit_tensor <= 0, torch.tensor(float('-inf')), torch.tensor(0.0)).cuda()
        if torch.all((unit_tensor>7) | (unit_tensor<=0)):
            action_type = torch.tensor(1).cuda()
        if torch.tensor(state["gold"][0]).float()<=0:
            action_type = torch.tensor(0).cuda()
        
        
        if(action_type==0):
            source_mask= source_mask + torch.where(unit_tensor == 20, torch.tensor(float('-inf')), torch.tensor(0.0))
        
        masked_source_logits = torch.clamp(source_tile_logits + source_mask, min=-1e9)
        
        
        source_tile_probs = torch.softmax((masked_source_logits).view(-1), dim=-1)
        source_tile_distribution = torch.distributions.Categorical(source_tile_probs)
        source_tile_idx = source_tile_distribution.sample()
        source_tile_idx_int = source_tile_idx.item()
        source_tile_q = source_tile_idx_int // env.size - env.game.size
        source_tile_r = source_tile_idx_int % env.size - env.game.size
        
        q_masking = source_tile_idx_int // env.size
        r_masking = source_tile_idx_int % env.size
        #print(q_masking)
        s_masking = -q_masking - r_masking
        
        q_coords, r_coords = torch.meshgrid(torch.arange(env.size), torch.arange(env.size), indexing='ij')
        
        
        dist = (torch.abs(q_coords - q_masking) + torch.abs(r_coords - r_masking) + torch.abs((-r_coords - q_coords) - (-r_masking - q_masking))) // 2
        dist = dist.cuda()
        if action_type == 1:
            condition = (dist == 1) 
        elif unit_tensor.squeeze(0)[q_masking][r_masking].item()==3:
            condition = (dist == 1) & (((unit_tensor == 0.0) & (land_water_tensor==1)) | (unit_tensor<0))
        else:
            condition = (dist>0) & (dist<=2) & (((unit_tensor == 0) & (land_water_tensor==0)) | (unit_tensor<0))
            
        target_mask = torch.where(condition, torch.tensor(0.0), torch.tensor(float('-inf'))).cuda()
        
        filled_mask = torch.where(unit_tensor > 0, torch.tensor(float('-inf')), torch.tensor(0.0))
        masked_target_logits = torch.clamp(target_tile_logits + target_mask  + terrain_mask + filled_mask, min=-1e9)
        
        target_tile_probs = torch.softmax((masked_target_logits).view(-1), dim=-1)
        target_tile_distribution = torch.distributions.Categorical(target_tile_probs)
        target_tile_idx = target_tile_distribution.sample()
        target_tile_idx_int = target_tile_idx.item()
        target_tile_q = target_tile_idx_int // env.size - env.game.size
        target_tile_r = target_tile_idx_int % env.size - env.game.size
        player = env.game.current_player_index
        
        next_state_this_pov, next_state_next_pov, reward, done, _ = env.step((action_type, source_tile_q, source_tile_r, target_tile_q, target_tile_r))
        episode_reward += reward
        
        
        if action_type == last_action:
            repeats +=1
        else:
            last_action = action_type
            repeats = 0
            
        #if repeats > 100:
        #    reward = reward -50
        
        if reward<0:
            invalid +=1
        
        next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).float().unsqueeze(0).cuda()
        next_gold_tensor = torch.tensor(next_state_this_pov["gold"]).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, _, _, next_value = target_model(next_grid_tensor, next_gold_tensor)
            td_target = reward + discount_factor * next_value * (1 - int(done))
        
        critic_loss = critic_loss_fn(value, td_target)
        
        advantage = (td_target - value).detach()
        
        log_prob_action_type = action_type_distribution.log_prob(action_type)
        log_prob_source_tile = source_tile_distribution.log_prob(source_tile_idx)
        log_prob_target_tile = target_tile_distribution.log_prob(target_tile_idx)
        
        actor_loss = -(log_prob_action_type + log_prob_source_tile + log_prob_target_tile) * I
        I = discount_factor * I
        
        total_loss = actor_loss * advantage + critic_loss
        
        total_optimizer.zero_grad()
        
        
        total_loss.backward()
        #print(advantage)
        for name, param in model.named_parameters():
            if param.grad is not None:
                if timestep % 2 ==0:
                    eligibility_traces_player1[name] = (discount_factor * trace_decay_rate * eligibility_traces_player1[name] + param.grad) 

                    param.grad = eligibility_traces_player1[name]
                else:
                    eligibility_traces_player2[name] = (discount_factor * trace_decay_rate * eligibility_traces_player2[name] + param.grad) 

                    param.grad = eligibility_traces_player2[name]
        
        #if initialized or victories > 4 or (victories > 0 and not done):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        if timestep % 100 == 0:
            target_model.load_state_dict(model.state_dict())
            
            
            
        total_optimizer.step()
        
        tau = 0.005
        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
            
        if done or (reward > 0 and epoch<100) or timestep % 100 == 0:
            grid = state["grid"][1]
            grid_str = '\n'.join([' '.join(map(str, row)) for row in grid])
            #new = next_state_next_pov["grid"][1]
            #new_str = '\n'.join([' '.join(map(str, row)) for row in new])
            os.system('cls')
            #print(source_mask)
            #print(env.game.atlas.get_hex(source_tile_q, source_tile_r, -source_tile_q - source_tile_r).unit is not None)
            sys.stdout.write(str(state["gold"]) + "\n")
            sys.stdout.flush
            sys.stdout.write(grid_str)
            sys.stdout.flush()
            sys.stdout.write(f"\n Step: {timestep}, player: {player}, reward: {reward}, Epoch: {epoch}, action: {(action_type.item(), source_tile_q, source_tile_r, target_tile_q, target_tile_r)}\nvictories: {victories}, value, next_value: {value.item(), next_value.item()} \ninvalid: {invalid}, avg_reward: {sum(cumulative_rewards) / len(cumulative_rewards)}" )
            sys.stdout.flush()
            #sys.stdout.write(new_str)
            #sys.stdout.flush()
            #time.sleep(2)
        state = next_state_next_pov
        if done:
            
            victories +=1
        
        elif timestep > 5000:                   #or repeats> 500: #idea for restarting instead of just punishing for repeats is it's stuck in a minimum,
                                                #so a new map may get it out of the distribution or whatnot
            done = True


    #per timestep is more informative.
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

