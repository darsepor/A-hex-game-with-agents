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
        (2, 250), 
        (3, 250),
        (4, 250),
        (5, 250)
    ]
    
    base_learning_rate = 0.001
    discount_factor = 0.99 #gamma
    trace_decay_rate = 0.95 #lambda
    initialized_curriculum_run = False 
    global_model_load_path = "overall_curriculum_actor_critic_model.pth" # Path for loading a model for the whole curriculum

    all_rewards_across_curriculum = []
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

    total_optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=0.01)
    #scheduler = optim.lr_scheduler.ExponentialLR(total_optimizer, gamma=0.99) doesn't make as much sense with the curriculum
    critic_loss_fn = nn.MSELoss()

    for stage_idx, (current_size, num_epochs_for_stage) in enumerate(curriculum_stages):
        print(f"--- Starting Curriculum Stage {stage_idx + 1}/{len(curriculum_stages)}: Size {current_size} for {num_epochs_for_stage} epochs ---")
        
        size = current_size 
        env = CustomGameEnv(size) # Initialize env with current_size for this stage
        

        stage_victories = 0
        cumulative_rewards_stage = deque(maxlen=10)
        
        rewards_per_timestep_stage = []

        for epoch in range(num_epochs_for_stage):
            episode_reward = 0
            eligibility_traces_player1 = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
            eligibility_traces_player2 = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
            
            state = env.reset(size) # Reset env with the correct current size for this episode
            done = False
            timestep = -1
            
            target_model.load_state_dict(model.state_dict())

            print(f"Stage {stage_idx+1} (Size {size}) - Epoch: {epoch+1}/{num_epochs_for_stage} (Global Ep: {global_episode_count +1})")
            
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
                
                episode_reward += reward
                
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
                    VIS_P1_SOLDIER = "S₁"
                    VIS_P1_BATTLESHIP = "B₁"
                    VIS_P1_CITY = "C₁"
                    VIS_P2_SOLDIER = "S₂"
                    VIS_P2_BATTLESHIP = "B₂"
                    VIS_P2_CITY = "C₂"
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
                    sys.stdout.write(f"Stage {stage_idx+1}, Size {size}, Ep {epoch+1}, Step: {timestep}, P: {player}, Step Reward: {reward:.2f}, Ep. Reward: {episode_reward:.2f}\n")
                    sys.stdout.write(f"Action: {(action_type.item(), coords['source_q'] + env.game.size, coords['source_r'] + env.game.size, coords['target_q'] + env.game.size, coords['target_r'] + env.game.size)}\n")
                    sys.stdout.write(f"Victories (stage): {stage_victories}, V_est: {value.item():.2f}, TD_target: {td_target.item():.2f}\n")
                    avg_rew_str = "N/A"
                    if cumulative_rewards_stage:
                        avg_rew_str = f"{sum(cumulative_rewards_stage) / len(cumulative_rewards_stage):.2f}"
                    sys.stdout.write(f"Avg Ep. Reward/Step (last 10): {avg_rew_str}\n")
                    sys.stdout.flush()
                
                state = next_state_next_pov
                
                if done:
                    if reward > 10:
                        stage_victories += 1
                elif timestep > 500:
                    done = True
            
            if timestep >= 0:
                avg_reward_this_episode = episode_reward / (timestep + 1)
            else:
                avg_reward_this_episode = 0
            
            cumulative_rewards_stage.append(avg_reward_this_episode)
            rewards_per_timestep_stage.append(avg_reward_this_episode)
            all_rewards_across_curriculum.append(avg_reward_this_episode)
            global_episode_count += 1

            #scheduler.step()


        torch.save(model.state_dict(), global_model_load_path) 
        print(f"Saved model checkpoint to {global_model_load_path} after stage {stage_idx + 1} (Size: {size})")

        if rewards_per_timestep_stage:
            # timestamp = time.strftime("%Y%m%d-%H%M%S") # Timestamp now in folder name
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(len(rewards_per_timestep_stage))), rewards_per_timestep_stage, label=f"Stage {stage_idx + 1} (Size {size}) Avg Reward/Timestep")
            plt.xlabel(f"Epoch in Stage {stage_idx + 1}")
            plt.ylabel("Average Reward per Timestep")
            plt.title(f"Rewards for Curriculum Stage {stage_idx + 1} - Size {size}")
            plt.legend(loc='best')
            plt.grid(True)
            # Use plots_dir_this_run and simpler filename
            stage_plot_filename = os.path.join(plots_dir_this_run, f"size-{size}-stage{stage_idx + 1}_rewards.png") 
            plt.savefig(stage_plot_filename)
            print(f"Saved stage plot to {stage_plot_filename}")
            plt.close()

    print("--- Curriculum Training Complete ---")

    # overall_rewards_file = f"{plots_dir}all_curriculum_rewards.json" # Already removed
    # with open(overall_rewards_file, "w") as f:
    #     json.dump(all_rewards_across_curriculum, f)

    # Remove the final global plot section
    # plt.figure(figsize=(12, 7))
    # plt.plot(list(range(len(all_rewards_across_curriculum))), all_rewards_across_curriculum, label="Avg Reward/Timestep per Episode")
    # 
    # current_ep_marker = 0
    # stage_labels_added = set()
    # for stage_idx_plot, (stage_size, num_epochs_in_stage_plot) in enumerate(curriculum_stages):
    #     if stage_idx_plot > 0 : 
    #         label = f'Start Stage {stage_idx_plot+1} (Size {stage_size})'
    #         if label in stage_labels_added: label = None 
    #         else: stage_labels_added.add(label)
    #         plt.axvline(x=current_ep_marker -0.5, color='r', linestyle='--', label=label)
    #     current_ep_marker += num_epochs_in_stage_plot
    #
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    # plt.xlabel("Global Episode Number")
    # plt.ylabel("Average Reward per Timestep")
    # plt.title("Aggregated Rewards Over Curriculum Training")
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.savefig(f"{plots_dir}curriculum_training_aggregated_rewards.png")
    # print(f"Saved aggregated curriculum rewards plot to {plots_dir}curriculum_training_aggregated_rewards.png")
    # plt.show()
    
if __name__ == "__main__":
    main()