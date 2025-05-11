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
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_single_episode(
    env, model, target_model, total_optimizer, critic_loss_fn, device,
    current_size, initial_state, discount_factor, trace_decay_rate, scheduler,
    log_display_info,
    max_episode_steps,
    log_conditionally
):
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    current_episode_total_reward = 0
    named_params_with_grad = {name: param for name, param in model.named_parameters() if param.requires_grad}
    eligibility_traces_player1 = {name: torch.zeros_like(param, device=device) for name, param in named_params_with_grad.items()}
    eligibility_traces_player2 = {name: torch.zeros_like(param, device=device) for name, param in named_params_with_grad.items()}
            
    state = initial_state
    done = False
    timestep = -1 
    
    last_reward = 0 

    while not done:
        timestep += 1
        grid_tensor = torch.tensor(state["grid"]).long().unsqueeze(0).to(device)
        gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).to(device)
        
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
        player_who_acted = env.game.current_player_index
        
        next_state_this_pov, next_state_next_pov, reward, done_flag, _ = env.step(
            (action_type, coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r']))
        
        current_episode_total_reward += reward
        last_reward = reward 
        done = done_flag 
        
        next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).long().unsqueeze(0).to(device)
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
        
        for name, param in named_params_with_grad.items():
            if param.grad is not None:
                current_trace_val = discount_factor * trace_decay_rate
                if player_who_acted == 0:
                    eligibility_traces_player1[name] = (current_trace_val * eligibility_traces_player1[name] + param.grad)
                    param.grad = eligibility_traces_player1[name]
                else: 
                    eligibility_traces_player2[name] = (current_trace_val * eligibility_traces_player2[name] + param.grad)
                    param.grad = eligibility_traces_player2[name]
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_optimizer.step()
        
        should_log = (not log_conditionally and timestep % 2 == 0) or (done or reward > 0 or timestep % 10 == 0)
        if should_log:
            current_id_grid = state["grid"]
            VIS_EMPTY_LAND, VIS_EMPTY_WATER = "🟩", "🟦"
            VIS_P1_SOLDIER, VIS_P1_BATTLESHIP, VIS_P1_CITY = "♘ ", "♗ ", "♔ "
            VIS_P2_SOLDIER, VIS_P2_BATTLESHIP, VIS_P2_CITY = "♞ ", "♝ ", "♚ "
            VIS_OUT_OF_BOUNDS, VIS_UNKNOWN = "⬛", "?"
            local_vis_map = {0: VIS_EMPTY_LAND, 1: VIS_EMPTY_WATER, 2: VIS_P1_SOLDIER, 3: VIS_P1_BATTLESHIP, 4: VIS_P1_CITY, 5: VIS_P2_SOLDIER, 6: VIS_P2_BATTLESHIP, 7: VIS_P2_CITY, 8: VIS_OUT_OF_BOUNDS}

            grid_str_rows = [ " ".join([local_vis_map.get(cell_id, VIS_UNKNOWN) for cell_id in row]) for row in current_id_grid ]
            grid_str = "\n".join(grid_str_rows)

            if os.name == 'nt': os.system('cls')
            else: os.system('clear')
            
            sys.stdout.write(f"Gold: {str(state['gold'])}\n")
            sys.stdout.write(grid_str + "\n")
            sys.stdout.write(f"{log_display_info['prefix']}, Step: {timestep}, P_acted: {player_who_acted}, Step Reward: {reward:.2f}, Ep. Total Reward: {current_episode_total_reward:.2f}\n")
            sys.stdout.write(f"Action: {(action_type.item(), coords['source_q'] + env.game.size, coords['source_r'] + env.game.size, coords['target_q'] + env.game.size, coords['target_r'] + env.game.size)}\n")
            sys.stdout.write(f"Victories (info): {log_display_info.get('victories_info', 'N/A')}, V_est: {value.item():.2f}, TD_target: {td_target.item():.2f}\n")
            sys.stdout.write(f"Avg of Ep. Avg Step Reward (last 10 eps): {log_display_info.get('avg_recent_reward_str', 'N/A')}\n")
            
            current_lr_for_display = scheduler.get_last_lr()[0]
            sys.stdout.write(f"Current LR: {current_lr_for_display:.7f}\n")
            sys.stdout.flush()
        
        state = next_state_next_pov 
        
        if done: 
            break 
        if timestep >= max_episode_steps: 
            done = True
            break
            
    episode_length = timestep + 1
    return current_episode_total_reward, episode_length, last_reward

def main():
    
    
    curriculum_stages = [
        (2, 100), 
        (3, 100),
        (4, 100),
        (5, 100)
    ]

    random_sizes_list = [2, 3, 4, 5]
    total_episodes_random_size = 400
    
    
    use_curriculum_training = False
    
    initial_learning_rate = 0.001
    lr_decay_gamma = 0.99
    discount_factor = 0.99
    trace_decay_rate = 0.8
    embedding_dim = 16
    max_episode_steps = 500
    
    log_conditionally = True

    load_initial_model = False
    global_model_path = "overall_actor_critic_model.pth"

    global_all_episode_avg_step_rewards_history = []
    global_episode_count = 0
    
    base_plots_dir = "./plots/"
    ensure_dir(base_plots_dir)
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    plots_dir_this_run = os.path.join(base_plots_dir, run_timestamp)
    ensure_dir(plots_dir_this_run)
    print(f"Saving plots for this run to: {plots_dir_this_run}")

    model = AttentionCNN(embedding_dim).to(device)
    target_model = AttentionCNN(embedding_dim).to(device)

    if load_initial_model and os.path.exists(global_model_path):
        print(f"Loading pre-trained model from: {global_model_path}")
        model.load_state_dict(torch.load(global_model_path, map_location=device))
    else:
        print("Initializing a new model.")

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    total_optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
    critic_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(total_optimizer, gamma=lr_decay_gamma)

    recent_episode_avg_step_rewards = deque(maxlen=10)

    if use_curriculum_training:
        print("--- Starting Curriculum Training ---")
        for stage_idx, (current_size, num_episodes_in_stage) in enumerate(curriculum_stages):
            current_lr_for_print = scheduler.get_last_lr()[0]
            print(f"--- Curriculum Stage {stage_idx + 1}/{len(curriculum_stages)}: Size {current_size} for {num_episodes_in_stage} episodes (LR: {current_lr_for_print:.7f}) ---")
            
            env = CustomGameEnv(current_size)
            stage_victories = 0
            current_stage_episode_avg_step_rewards_history = []

            for episode_in_stage in range(num_episodes_in_stage):
                initial_state = env.reset(current_size)
                
                avg_recent_reward_str = "N/A"
                if recent_episode_avg_step_rewards:
                    avg_val = sum(recent_episode_avg_step_rewards) / len(recent_episode_avg_step_rewards)
                    avg_recent_reward_str = f"{avg_val:.2f}"

                log_display_info = {
                    'prefix': f"Stage {stage_idx+1}, Size {current_size}, Ep {episode_in_stage+1}/{num_episodes_in_stage}",
                    'victories_info': f"{stage_victories}",
                    'avg_recent_reward_str': avg_recent_reward_str
                }
                
                episode_total_reward, episode_length, last_reward = run_single_episode(
                    env, model, target_model, total_optimizer, critic_loss_fn, device,
                    current_size, initial_state, discount_factor, trace_decay_rate, scheduler,
                    log_display_info,
                    max_episode_steps,
                    log_conditionally
                )

                if last_reward > 10:
                    stage_victories += 1
                
                episode_avg_step_reward = 0.0
                if episode_length > 0:
                    episode_avg_step_reward = episode_total_reward / episode_length
                
                recent_episode_avg_step_rewards.append(episode_avg_step_reward)
                current_stage_episode_avg_step_rewards_history.append(episode_avg_step_reward)
                global_all_episode_avg_step_rewards_history.append(episode_avg_step_reward)
                global_episode_count += 1
                scheduler.step()

            torch.save(model.state_dict(), global_model_path) 
            print(f"Saved model checkpoint to {global_model_path} after stage {stage_idx + 1} (Size: {current_size})")

            if current_stage_episode_avg_step_rewards_history:
                plt.figure(figsize=(10, 6))
                plt.plot(current_stage_episode_avg_step_rewards_history, label=f"Stage {stage_idx + 1} (Size {current_size}) Avg Reward/Timestep")
                plt.xlabel(f"Episode in Stage {stage_idx + 1}")
                plt.ylabel("Average Reward per Timestep in Episode")
                plt.title(f"Rewards for Curriculum Stage {stage_idx + 1} - Size {current_size}")
                plt.legend(loc='best')
                stage_plot_filename = os.path.join(plots_dir_this_run, f"size-{current_size}-stage{stage_idx + 1}_rewards.png") 
                plt.savefig(stage_plot_filename)
                print(f"Saved stage plot to {stage_plot_filename}")
                plt.close()
        
        print("--- Curriculum Training Complete ---")
        torch.save(model.state_dict(), global_model_path)
        print(f"Saved final model to {global_model_path} after curriculum.")

    else: 
        print(f"--- Starting Random Size Training for {total_episodes_random_size} episodes ---")
        print(f"Sampling sizes from: {random_sizes_list}")
        overall_victories = 0
        
        for episode_num in range(total_episodes_random_size):
            current_size = random.choice(random_sizes_list)
            env = CustomGameEnv(current_size)
            initial_state = env.reset(current_size)

            avg_recent_reward_str = "N/A"
            if recent_episode_avg_step_rewards:
                avg_val = sum(recent_episode_avg_step_rewards) / len(recent_episode_avg_step_rewards)
                avg_recent_reward_str = f"{avg_val:.2f}"

            log_display_info = {
                'prefix': f"Random Training Ep {episode_num+1}/{total_episodes_random_size}, Size {current_size}",
                'victories_info': f"{overall_victories}",
                'avg_recent_reward_str': avg_recent_reward_str
            }

            episode_total_reward, episode_length, last_reward = run_single_episode(
                env, model, target_model, total_optimizer, critic_loss_fn, device,
                current_size, initial_state, discount_factor, trace_decay_rate, scheduler,
                log_display_info,
                max_episode_steps,
                log_conditionally
            )

            if last_reward > 10:
                overall_victories += 1
            
            episode_avg_step_reward = 0.0
            if episode_length > 0:
                episode_avg_step_reward = episode_total_reward / episode_length
            
            recent_episode_avg_step_rewards.append(episode_avg_step_reward)
            global_all_episode_avg_step_rewards_history.append(episode_avg_step_reward)
            global_episode_count += 1
            scheduler.step()

            if (episode_num + 1) % 100 == 0 or episode_num == total_episodes_random_size - 1:
                print(f"Random Training: Episode {episode_num + 1}/{total_episodes_random_size}, Size: {current_size}, Victories: {overall_victories}, Avg Recent Reward: {avg_recent_reward_str}, LR: {scheduler.get_last_lr()[0]:.7f}")

        print("--- Random Size Training Complete ---")
        torch.save(model.state_dict(), global_model_path)
        print(f"Saved model to {global_model_path} after random size training.")

    if global_all_episode_avg_step_rewards_history:
        plt.figure(figsize=(12, 7))
        plt.plot(global_all_episode_avg_step_rewards_history, label="Global Avg Reward/Timestep per Episode")
        plt.xlabel("Global Episode Count")
        plt.ylabel("Average Reward per Timestep in Episode")
        plt.title(f"Overall Training Performance ({'Curriculum' if use_curriculum_training else 'Random Sizes'}) - Run: {run_timestamp}")
        plt.legend(loc='best')
        global_plot_filename = os.path.join(plots_dir_this_run, f"global_training_rewards_{run_timestamp}.png")
        plt.savefig(global_plot_filename)
        print(f"Saved global training plot to {global_plot_filename}")
        plt.close()
    
    print(f"All plots and models for run {run_timestamp} saved in {plots_dir_this_run} and {global_model_path} respectively.")

if __name__ == "__main__":
    main()