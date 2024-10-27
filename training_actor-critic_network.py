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
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, num_action_types):
        self.input_shape = input_shape
        linear = 512
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Flatten()
        
        self.shared_fc = nn.Linear(64 * input_shape[1] * input_shape[2] + 2, linear)  #Shared layer after convolution
        #print(self.shared_fc)
        self.action_type_head = nn.Linear(linear, num_action_types)
        self.source_tile_head = nn.ConvTranspose2d(64 + 2 + 16, 1, kernel_size=3, stride=1, padding=1)
        self.target_tile_head = nn.ConvTranspose2d(64 + 2 + 1 + 16, 1, kernel_size=3, stride=1, padding=1) #we dont give a pathing scaffolding,
                                                                                             #nor do we mask invalid actions,
                                                                                             #so logits much be close
                                                                                             #update: had to use maybe too many masks to barely work
                                                                                             #(feels like "might as well write a bunch of if statements"), better approach needed
                                                                                             #update2: maybe an even more extensive/adaptive mask and rewarding only
                                                                                             #winning and nothing else would still produce something interesting 
                                                                                             #(ie, focusing on strategy instead of picking out "valid" actions),
                                                                                             #intermediate rewards depending on how many pieces player has could help with the 'sparsity'
                                                                                             #also, could maybe train against (or to imitate) an improved SimpleAI
                                                                                             #and punish for losing against it
                                                                                             #update3: made game more strategic instead of a stalemate slog by adding an economy,
                                                                                             #masking being overhauled is planned next as even now most actions are invalid. 
                                                                                             #Also, fixed a massive oversight of the next_state the
                                                                                             #network sees being from the POV of the opponent.
        self.critic_fc = nn.Linear(linear, 1)                                                #I'm going off by the proportion of how many games get resolved within a 1000 timesteps--
                                                                                             #at convergence and the best performance so far is around 30%. Getting better!
        

    def forward(self, grid, gold):
        x1 = torch.relu(self.conv1(grid))
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2)) #skip connections or separating actor and critic
                                        #give unimpressive 20 epochs
        
        shared_features = x3
        x = self.fc(x3)
        
        x = torch.cat((x, gold), dim=1)
        x = torch.relu(self.shared_fc(x))
        
        action_type_logits = self.action_type_head(x)
        
        action_info = action_type_logits.unsqueeze(-1).unsqueeze(-1)
        action_info = action_info.expand(-1, 2, self.input_shape[1], self.input_shape[2])
        #print(action_info.shape)
        combined_source_input = torch.cat((shared_features, action_info, x1), dim=1)
        source_tile_logits = self.source_tile_head(combined_source_input)
        
        combined_target_input = torch.cat((combined_source_input, source_tile_logits), dim=1)
        target_tile_logits = self.target_tile_head(combined_target_input)
        
        value = self.critic_fc(x)

        return action_type_logits, source_tile_logits.squeeze(1), target_tile_logits.squeeze(1), value #further masking of target logits below






epochs = 500
learning_rate = 0.001
discount_factor = 0.99 #gamma
trace_decay_rate = 0.6 #lambda
initialized = False #whether we're not training from scratch

size = 5 #shared linear layer prevents from generalizing in diff sizes 
         #edit: oh right, 
         # 
         # just reuse convolutional layers in the initialization

env = CustomGameEnv(size)
#print(env.size)
model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()

#saved_state_dict = torch.load("size-5-actor_critic_model.pth")
#model.load_state_dict(saved_state_dict)
#initialized = True
eligibility_traces_player1 = {name: torch.zeros_like(param, device='cuda') for name, param in model.named_parameters()}
eligibility_traces_player2 = {name: torch.zeros_like(param, device='cuda') for name, param in model.named_parameters()}

total_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.01)

critic_loss_fn = nn.MSELoss()

victories = 0
for epoch in range(epochs):
    #size = size + min(10, epoch // 2)
    #model.input_shape = (2, size, size)
    state = env.reset(size)
    done = False
    total_reward = 0
    timestep = -1
    print(f"Epoch: {epoch}")
    recent_actions = deque(maxlen=500)
    
    last_action = 0
    repeats = 0
    I = 1
    
    while not done:
        timestep +=1
        grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).cuda()
        gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).cuda()
        unit_tensor = torch.tensor(state["grid"][1]).float().unsqueeze(0).cuda()
        
        terrain_mask = env.mask.cuda()
        
        source_mask = torch.where(unit_tensor <= 0, torch.tensor(float('-inf')), torch.tensor(0.0)).cuda()
        
        action_type_logits, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
        temperature = 2.0
        #if (timestep < 1):
        #    action_type_logits = torch.tensor([5, 0])
        action_type_logits = action_type_logits
        action_type_probs = torch.softmax(action_type_logits, dim=-1)
        action_type_distribution = torch.distributions.Categorical(action_type_probs)
        action_type = action_type_distribution.sample()
        #if torch.all((unit_tensor <= 0) | (unit_tensor > 7)):
        #    action_type = torch.tensor(1).cuda() feels wrong. with only 2 action types 
        # maybe you should get punished if you get the logits wrong -- a fundamental misunderstanding of the game board in a binary way
        
            
        
        
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
        condition = (dist > 0) & (dist <= 2) 
        #unit_tensor_cpu = unit_tensor.cpu()
        if action_type == 1 or unit_tensor.squeeze(0)[q_masking][r_masking].item()==3:
            condition = (dist == 1) 
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
        
        if action_type == last_action:
            repeats +=1
        else:
            last_action = action_type
            repeats = 0
            
        if repeats > 100:
            reward = reward -50
        
        next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).float().unsqueeze(0).cuda()
        next_gold_tensor = torch.tensor(next_state_this_pov["gold"]).float().unsqueeze(0).cuda()
        
        with torch.no_grad():
            _, _, _, next_value = model(next_grid_tensor, next_gold_tensor)
            td_target = reward + discount_factor * next_value * (1 - int(done))
        
        critic_loss = critic_loss_fn(value, td_target)
        
        advantage = (td_target - value).detach()
        
        log_prob_action_type = action_type_distribution.log_prob(action_type)
        log_prob_source_tile = source_tile_distribution.log_prob(source_tile_idx)
        log_prob_target_tile = target_tile_distribution.log_prob(target_tile_idx) #a bit iffy, though maybe it is better to have a more
                                                                                  #generalizable logit matrices that work with many board states instead of
                                                                                  #trying to learn a ton of differing validities for differing states
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
            
            
        total_optimizer.step()
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
            sys.stdout.write(f"\n Step: {timestep}, player: {player}, reward: {reward}, Epoch: {epoch}, action: {(action_type.item(), source_tile_q, source_tile_r, target_tile_q, target_tile_r)}\nvictories: {victories}, value, next_value: {value.item(), next_value.item()}")
            sys.stdout.flush()
            #sys.stdout.write(new_str)
            #sys.stdout.flush()
            #time.sleep(2)
        state = next_state_next_pov
        if done:
            
            victories +=1
            time.sleep(5)
        
        elif repeats > 500 or timestep > 1000: #idea for restarting instead of just punishing for repeats is it's stuck in a minimum,
                                                #so a new map may get it out of the distribution or whatnot
            done = True
    #if epoch % 10 == 0:
    #    print(f"Ep Reward: {total_reward}")

torch.save(model.state_dict(), f"size-{size}-actor_critic_model.pth")

print("Training complete!!")
