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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    curriculum_stages = [
        (2, 20), 
        (3, 20),
        (4, 20),
        (5, 20)
    ]
    
    stage_initial_learning_rates = [0.001, 0.0008, 0.0006, 0.0004] 
    if len(stage_initial_learning_rates) != len(curriculum_stages):
        raise ValueError("stage_initial_learning_rates must have the same number of elements as curriculum_stages.")

    #within_stage_lr_decay_gamma = 0.99 # Decay factor for ExponentialLR within a stage

    discount_factor = 0.99 #gamma
    trace_decay_rate = 0.95 #lambda, like 20 steps credit window
    initialized_curriculum_run = False 
    global_model_load_path = "overall_curriculum_actor_critic_model.pth" # Path for loading a model for the whole curriculum

    global_all_episode_avg_step_rewards_history = []
    global_episode_count = 0
    
    # Create a timestamped subdirectory for this run's plots
    base_plots_dir = "./plots/"
    ensure_dir(base_plots_dir) # Ensure the base ./plots directory exists
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    plots_dir_this_run = os.path.join(base_plots_dir, run_timestamp)
    ensure_dir(plots_dir_this_run)
    print(f"Saving plots for this run to: {plots_dir_this_run}")

 
    initial_size_for_model_init = curriculum_stages[0][0]
    # Embedding dimension (same as network input channels after embedding)
    EMBEDDING_DIM = 16

    # Instantiate network with embedding_dim as input channels
    model = ResActorCriticNetwork((EMBEDDING_DIM, initial_size_for_model_init, initial_size_for_model_init), 2).to(device)
    target_model = ResActorCriticNetwork((EMBEDDING_DIM, initial_size_for_model_init, initial_size_for_model_init), 2).to(device)

    if initialized_curriculum_run and os.path.exists(global_model_load_path):
        print(f"Loading pre-trained curriculum model from: {global_model_load_path}")
        model.load_state_dict(torch.load(global_model_load_path, map_location=device))
    else:
        print("Initializing a new model for the curriculum.")

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    total_optimizer = optim.AdamW(model.parameters(), lr=stage_initial_learning_rates[0], weight_decay=0.01)
    critic_loss_fn = nn.MSELoss()
    #scheduler = None # Will be initialized at the start of each stage

    for stage_idx, (current_size, num_episodes_in_stage) in enumerate(curriculum_stages):
        current_initial_lr = stage_initial_learning_rates[stage_idx]
        for param_group in total_optimizer.param_groups:
            param_group['lr'] = current_initial_lr
        
        # Initialize a new scheduler for this stage
        #scheduler = optim.lr_scheduler.ExponentialLR(total_optimizer, gamma=within_stage_lr_decay_gamma)
        
        print(f"--- Starting Curriculum Stage {stage_idx + 1}/{len(curriculum_stages)}: Size {current_size} for {num_episodes_in_stage} episodes (Initial LR: {current_initial_lr}) ---")
        
        size = current_size 
        env = CustomGameEnv(size) # Initialize env with current_size for this stage
        

        stage_victories = 0
        recent_episode_avg_step_rewards = deque(maxlen=10)
        
        current_stage_episode_avg_step_rewards_history = []

        for episode_in_stage in range(num_episodes_in_stage):
            current_episode_total_reward = 0
            eligibility_traces_player1 = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
            eligibility_traces_player2 = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
            
            steps_since_last_positive_reward_p1 = 0
            steps_since_last_positive_reward_p2 = 0
            
            state = env.reset(size)
            done = False
            timestep = -1
            
            target_model.load_state_dict(model.state_dict())

            print(f"Stage {stage_idx+1} (Size {size}) - Episode: {episode_in_stage+1}/{num_episodes_in_stage} (Global Ep: {global_episode_count +1})") # Renamed
            
            while not done:
                timestep += 1
                # grid_tensor is integer ID grid: use long dtype
                grid_tensor = torch.tensor(state["grid"]).long().unsqueeze(0).to(device)
                gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).to(device)
                
                # The same model instance handles varying tensor sizes from the env
                action_values, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
                
                sampling_results = env.sample_apply_masks(action_values, source_tile_logits, target_tile_logits, state, device)
                
                if sampling_results is None: 
                    env.game.next_turn()
                    state = env._get_observation()
                    continue
                
                action_type = sampling_results['action_type']
                source_tile_idx = sampling_results['source_tile_idx']
                target_tile_idx = sampling_results['target_tile_idx']
                coords = sampling_results['coordinates']
                player = env.game.current_player_index
                
                next_state_this_pov, next_state_next_pov, reward, done, _ = env.step(
                    (action_type, coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r']))
                
                original_env_reward = reward
                
                if player == 0:
                    if original_env_reward > 0:
                        steps_since_last_positive_reward_p1 = 0
                    else:
                        steps_since_last_positive_reward_p1 += 1
                        if steps_since_last_positive_reward_p1 >= 15:
                            reward += -0.01
                elif player == 1:
                    if original_env_reward > 0:
                        steps_since_last_positive_reward_p2 = 0
                    else:
                        steps_since_last_positive_reward_p2 += 1
                        if steps_since_last_positive_reward_p2 >= 15:
                            reward += -0.01
                
                current_episode_total_reward += reward
                
                next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).float().unsqueeze(0).to(device)
                next_gold_tensor = torch.tensor(next_state_this_pov["gold"]).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    _, _, _, next_value = target_model(next_grid_tensor, next_gold_tensor)
                    td_target = reward + discount_factor * next_value * (1 - int(done))
                
                critic_loss = critic_loss_fn(value, td_target)
                advantage = (td_target - value).detach()
                
                log_prob_action_type = sampling_results['action_type_distribution'].log_prob(action_type.float())
                log_prob_source_tile = sampling_results['source_tile_distribution'].log_prob(source_tile_idx)
                log_prob_target_tile = sampling_results['target_tile_distribution'].log_prob(target_tile_idx)
                
                actor_loss = -(log_prob_action_type + log_prob_source_tile + log_prob_target_tile) 
                
                total_loss = actor_loss * advantage + critic_loss
                
                total_optimizer.zero_grad()
                total_loss.backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        current_trace_val = discount_factor * trace_decay_rate
                        if timestep % 2 == 0:
                            eligibility_traces_player1[name] = (current_trace_val * eligibility_traces_player1[name] + param.grad)
                            param.grad = eligibility_traces_player1[name]
                        else:
                            eligibility_traces_player2[name] = (current_trace_val * eligibility_traces_player2[name] + param.grad)
                            param.grad = eligibility_traces_player2[name]
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_optimizer.step()
                
                if done or reward > 0 or timestep % 10 == 0:
                    # Visualization based on the new ID grid
                    current_id_grid = state["grid"]


                    VIS_EMPTY_LAND = "🟩"
                    VIS_EMPTY_WATER = "🟦"
                    VIS_P1_SOLDIER = "♘ "
                    VIS_P1_BATTLESHIP = "♗ "
                    VIS_P1_CITY = "♔ "
                    VIS_P2_SOLDIER = "♞ "
                    VIS_P2_BATTLESHIP = "♝ "
                    VIS_P2_CITY = "♚ "
                    VIS_OUT_OF_BOUNDS = "⬛" # For out-of-bounds cells
                    VIS_UNKNOWN = "?" # For any IDs not explicitly mapped

                    local_vis_map = {
                        0: VIS_EMPTY_LAND,    # EMPTY_LAND
                        1: VIS_EMPTY_WATER,   # EMPTY_WATER
                        2: VIS_P1_SOLDIER,    # P1_SOLDIER
                        3: VIS_P1_BATTLESHIP, # P1_BATTLESHIP
                        4: VIS_P1_CITY,       # P1_CITY
                        5: VIS_P2_SOLDIER,    # P2_SOLDIER
                        6: VIS_P2_BATTLESHIP, # P2_BATTLESHIP
                        7: VIS_P2_CITY,       # P2_CITY
                        8: VIS_OUT_OF_BOUNDS  # OUT_OF_BOUNDS
                    }

                    grid_str_rows = []
                    for row in current_id_grid:
                        row_str = []
                        for cell_id in row:
                            row_str.append(local_vis_map.get(cell_id, VIS_UNKNOWN))
                        grid_str_rows.append(" ".join(row_str))
                    grid_str = "\n".join(grid_str_rows)

                    if os.name == 'nt': os.system('cls')
                    else: os.system('clear')
                    
                    sys.stdout.write(f"Gold: {str(state['gold'])}\n")
                    sys.stdout.write(grid_str + "\n")
                    sys.stdout.write(f"Stage {stage_idx+1}, Size {size}, Ep {episode_in_stage+1}, Step: {timestep}, P: {player}, Step Reward: {reward:.2f}, Ep. Total Reward: {current_episode_total_reward:.2f}\n") # Clarified Ep. Reward
                    sys.stdout.write(f"Action: {(action_type.item(), coords['source_q'] + env.game.size, coords['source_r'] + env.game.size, coords['target_q'] + env.game.size, coords['target_r'] + env.game.size)}\n")
                    sys.stdout.write(f"Victories (stage): {stage_victories}, V_est: {value.item():.2f}, TD_target: {td_target.item():.2f}\n")
                    
                    avg_recent_avg_step_reward_str = "N/A"
                    if recent_episode_avg_step_rewards:
                        avg_recent_avg_step_reward = sum(recent_episode_avg_step_rewards) / len(recent_episode_avg_step_rewards)
                        avg_recent_avg_step_reward_str = f"{avg_recent_avg_step_reward:.2f}"
                    sys.stdout.write(f"Avg of Ep. Avg Step Reward (last 10 eps): {avg_recent_avg_step_reward_str}\n")
                    sys.stdout.flush()
                
                state = next_state_next_pov
                
                if done:
                    if reward > 10:
                        stage_victories += 1
                elif timestep > 500:
                    done = True
            
            episode_length = timestep + 1
            current_episode_avg_step_reward = 0.0
            if episode_length > 0:
                current_episode_avg_step_reward = current_episode_total_reward / episode_length
            
            recent_episode_avg_step_rewards.append(current_episode_avg_step_reward)
            current_stage_episode_avg_step_rewards_history.append(current_episode_avg_step_reward)
            global_all_episode_avg_step_rewards_history.append(current_episode_avg_step_reward)
            global_episode_count += 1

            #scheduler.step()

        torch.save(model.state_dict(), global_model_load_path) 
        print(f"Saved model checkpoint to {global_model_load_path} after stage {stage_idx + 1} (Size: {size})")

        if current_stage_episode_avg_step_rewards_history:
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(len(current_stage_episode_avg_step_rewards_history))), current_stage_episode_avg_step_rewards_history, label=f"Stage {stage_idx + 1} (Size {size}) Avg Reward/Timestep per Episode") # Renamed list, updated label
            plt.xlabel(f"Episode in Stage {stage_idx + 1}")
            plt.ylabel("Average Reward per Timestep in Episode")
            plt.title(f"Rewards for Curriculum Stage {stage_idx + 1} - Size {size}")
            plt.legend(loc='best')
            stage_plot_filename = os.path.join(plots_dir_this_run, f"size-{size}-stage{stage_idx + 1}_rewards.png") 
            plt.savefig(stage_plot_filename)
            print(f"Saved stage plot to {stage_plot_filename}")
            plt.close()

    print("--- Curriculum Training Complete ---")

   
    
if __name__ == "__main__":
    main()