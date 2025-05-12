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
from res_net_AC import ResActorCriticNetwork
from attn_cnn import AttentionCNN
from local_hex_attention_transformer import HexTransformer
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_data, reward, next_state, done):
        self.buffer.append((state, action_data, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def sample_trajectory(self, length):
        if len(self.buffer) < length:
            return None
        start_index = random.randint(0, len(self.buffer) - length)
        return list(self.buffer)[start_index : start_index + length]

    def __len__(self):
        return len(self.buffer)

def run_single_episode(
    env, model, opponent_model, buffer, device,
    current_size, initial_state,
    log_display_info,
    max_episode_steps,
    player_to_train = 0
):
    """Runs one episode, simulating a full turn (P0 then P1) in each loop.
       Stores transitions (s0, a0, r0, s1, done1) in the buffer for `player_to_train`.
       Returns nothing as reward/trajectory data is not used by caller.
    """
    state = initial_state 
    done = False
    timestep = -1 

    while not done:
        timestep += 1
        if timestep >= max_episode_steps:
            break

        # --- Player 0's Turn (TrainNet) ---
        if env.game.current_player_index != player_to_train:
             break

        state_p0 = state 
        grid_tensor_p0 = torch.tensor(state_p0["grid"]).long().unsqueeze(0).to(device)
        gold_tensor_p0 = torch.tensor(state_p0["gold"]).float().unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            action_values_p0, source_logits_p0, target_logits_p0, _ = model(grid_tensor_p0, gold_tensor_p0)

        # Call sample_apply_masks without force args for sampling
        sampling_results_p0 = env.sample_apply_masks(action_values_p0, source_logits_p0, target_logits_p0, state_p0, device)

        action_p0_coords = None
        action_data_p0_buffer = None 

        if sampling_results_p0 is None: 
            env.game.next_turn()
            next_state_pov_opponent = env._get_observation()
            reward_p0 = 0.0
            done_p0 = env.game.is_game_over()
        else:
            # Extract sampled indices for env.step()
            action_p0_type = sampling_results_p0['action_type'] 
            source_tile_idx_p0 = sampling_results_p0['source_tile_idx'] 
            target_tile_idx_p0 = sampling_results_p0['target_tile_idx'] 
            
            action_p0_coords = (action_p0_type.item(), 
                                sampling_results_p0['coordinates']['source_q'],
                                sampling_results_p0['coordinates']['source_r'],
                                sampling_results_p0['coordinates']['target_q'],
                                sampling_results_p0['coordinates']['target_r'])
            
            # Get masks/dict for storage
            valid_source_mask_p0 = sampling_results_p0['valid_source_mask']
            valid_target_mask_p0 = sampling_results_p0['valid_target_mask']
            possible_actions_dict_p0 = sampling_results_p0['possible_actions_for_target']

            # Store INDICES and MASKS/DICT in buffer
            action_data_p0_buffer = {
                 'action_type': action_p0_type.cpu(), # Store index
                 'source_tile_idx': source_tile_idx_p0.cpu(), # Store index
                 'target_tile_idx': target_tile_idx_p0.cpu(), # Store index
                 # 'coordinates' field removed as it's derivable from indices if needed elsewhere, keep buffer lean
                 'valid_source_mask': valid_source_mask_p0.cpu(), # Store mask
                 'valid_target_mask': valid_target_mask_p0.cpu(), # Store mask
                 'possible_actions_for_target': possible_actions_dict_p0 # Store dict
            }
            _, next_state_pov_opponent, reward_p0, done_p0, _ = env.step(action_p0_coords)

        # --- Opponent's Turn (Player 1 / OpponentNet) ---
        next_state_pov_p0_final = None 
        done_final = done_p0      

        if not done_p0:
            if env.game.current_player_index == player_to_train:
                 break

            state_p1 = next_state_pov_opponent
            grid_tensor_p1 = torch.tensor(state_p1["grid"]).long().unsqueeze(0).to(device)
            gold_tensor_p1 = torch.tensor(state_p1["gold"]).float().unsqueeze(0).to(device)

            opponent_model.eval()
            with torch.no_grad():
                action_values_p1, source_logits_p1, target_logits_p1, _ = opponent_model(grid_tensor_p1, gold_tensor_p1)

            sampling_results_p1 = env.sample_apply_masks(action_values_p1, source_logits_p1, target_logits_p1, state_p1, device)

            if sampling_results_p1 is None: 
                env.game.next_turn()
                next_state_pov_p0_final = env._get_observation() 
                done_p1 = env.game.is_game_over()
            else:
                action_p1_type = sampling_results_p1['action_type'].item()
                action_p1_coords = (action_p1_type,
                                    sampling_results_p1['coordinates']['source_q'],
                                    sampling_results_p1['coordinates']['source_r'],
                                    sampling_results_p1['coordinates']['target_q'],
                                    sampling_results_p1['coordinates']['target_r'])

                _, next_state_pov_p0_final, _, done_p1, _ = env.step(action_p1_coords)

            done_final = done_p1

        if action_data_p0_buffer is not None:
            buffer.push(state_p0, action_data_p0_buffer, reward_p0, next_state_pov_p0_final, done_final)

        state = next_state_pov_p0_final
        done = done_final

    return

def train_from_buffer(
    model, target_model, buffer, total_optimizer, critic_loss_fn, device,
    batch_size, discount_factor, n_steps=20
):
    if len(buffer) < batch_size or len(buffer) < n_steps: 
        return None, None 

    model.train() 
    target_model.eval() 

    total_actor_loss = 0
    total_critic_loss = 0
    updates = 0

    indices = random.sample(range(len(buffer) - n_steps + 1), batch_size)

    for i in indices:
        trajectory = list(buffer.buffer)[i : i + n_steps]
        state_t, action_data_t, _, _, _ = trajectory[0]
        
        # N-Step Return Calculation, weird syntax the else execute if the for loop is not broken to bootstrap value estimate.
        n_step_return = 0.0
        for k in range(n_steps):
            _, _, reward_k, _, done_k = trajectory[k]
            n_step_return += (discount_factor ** k) * reward_k
            if done_k:
                break
        else: 
            _, _, _, state_t_n, _ = trajectory[n_steps - 1] 
            if state_t_n is not None: 
                grid_t_n = torch.tensor(state_t_n["grid"]).long().unsqueeze(0).to(device)
                gold_t_n = torch.tensor(state_t_n["gold"]).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, _, value_t_n = target_model(grid_t_n, gold_t_n)
                n_step_return += (discount_factor ** n_steps) * value_t_n.item()
        
        td_target = torch.tensor([n_step_return], dtype=torch.float32, device=device)

        # Forward pass with CURRENT model for state s_t
        grid_t = torch.tensor(state_t["grid"]).long().unsqueeze(0).to(device)
        gold_t = torch.tensor(state_t["gold"]).float().unsqueeze(0).to(device)
        
        current_action_logits, current_source_logits, current_target_logits, value_t = model(grid_t, gold_t)

        # Critic Loss
        critic_loss = critic_loss_fn(value_t.squeeze(), td_target)
        advantage = (td_target - value_t).detach()

        # Actor Loss
        # Retrieve stored action data (indices and masks/dict)
        action_data_t = trajectory[0][1] # Get action_data dict from the first step
        action_type_taken = action_data_t['action_type']
        source_tile_idx_taken = action_data_t['source_tile_idx']
        target_tile_idx_taken = action_data_t['target_tile_idx']
        # Masks/dict are implicitly passed via action_data_t if sample_apply_masks needs them, but we don't use them directly here.

        # Call sample_apply_masks in "forced" mode to get distributions based on current model outputs
        # Note: We pass the original state_t here
        forced_results = env.sample_apply_masks(
            current_action_logits, 
            current_source_logits, 
            current_target_logits, 
            state_t, 
            device,
            force_source_idx=source_tile_idx_taken,
            force_target_idx=target_tile_idx_taken,
            force_action_type=action_type_taken
        )

        # Check if forced calculation succeeded (it shouldn't fail if buffer data is valid)
        if forced_results is None:
            raise ValueError("env.sample_apply_masks failed during forced calculation in training. Check buffer data validity.")

        # Extract the distributions calculated using the CURRENT model outputs, but FORCED indices
        action_type_dist = forced_results['action_type_distribution']
        source_dist = forced_results['source_tile_distribution']
        target_dist = forced_results['target_tile_distribution']

        # Calculate log probabilities using the distributions and the STORED action indices
        # Need .to(device) again as they might have been stored on CPU
        log_prob_action_type = action_type_dist.log_prob(action_type_taken.to(device).float()) # Bernoulli needs float
        log_prob_source_tile = source_dist.log_prob(source_tile_idx_taken.to(device).long())
        log_prob_target_tile = target_dist.log_prob(target_tile_idx_taken.to(device).long())

        # --- Calculate Actor Loss --- 
        actor_loss = -(log_prob_action_type + log_prob_source_tile + log_prob_target_tile)
        weighted_actor_loss = actor_loss * advantage.squeeze()

        # Total Loss and Update
        total_loss = weighted_actor_loss + critic_loss

        total_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_optimizer.step()
        
        total_actor_loss += weighted_actor_loss.item()
        total_critic_loss += critic_loss.item()
        updates += 1

    if updates > 0:
        avg_actor_loss = total_actor_loss / updates
        avg_critic_loss = total_critic_loss / updates
        return avg_actor_loss, avg_critic_loss
    else:
        return None, None

def main():
    random_sizes_list = [2, 3, 4, 5, 6]
    total_episodes = 10000 
    
    initial_learning_rate = 0.0003
    lr_decay_gamma = 0.998
    discount_factor = 0.99
    n_steps = 20 
    replay_buffer_capacity = 100000 
    batch_size = 128 
    training_interval = 1 
    target_update_freq = 200 
    opponent_update_freq = 500 
    max_episode_steps = 600 
    save_model_freq = 500 

    global_model_path = "td_20_rand_actor_critic_model.pth"

    training_steps = 0
    
    model = HexTransformer().to(device)
    target_model = HexTransformer().to(device)
    opponent_model = HexTransformer().to(device) 

    opponent_model.load_state_dict(model.state_dict())
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    opponent_model.eval() 

    total_optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
    critic_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(total_optimizer, gamma=lr_decay_gamma)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    for episode_num in range(total_episodes):
        size = random.choice(random_sizes_list)
        env = CustomGameEnv(size=size)
        initial_state = env.reset()

        run_single_episode(
            env, model, opponent_model, replay_buffer, device, size, initial_state, 
            {}, max_episode_steps, player_to_train=0
        )

        if len(replay_buffer) >= batch_size + n_steps: 
            model.train() 
            avg_actor_loss, avg_critic_loss = train_from_buffer(
                model, target_model, replay_buffer, total_optimizer, critic_loss_fn, device, 
                batch_size, discount_factor, n_steps
            )
            model.eval() 
                
            if avg_actor_loss is not None:
                training_steps += 1
                if training_steps % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                scheduler.step() 
        
        current_episode_index_one_based = episode_num + 1
        if current_episode_index_one_based > 0 and current_episode_index_one_based % opponent_update_freq == 0:
            opponent_model.load_state_dict(model.state_dict())
            opponent_model.eval() 
        
        if current_episode_index_one_based > 0 and current_episode_index_one_based % save_model_freq == 0:
            torch.save(model.state_dict(), global_model_path)

if __name__ == "__main__":
    main()
