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
from CNNAC import ActorCriticNetwork




def sample_apply_masks(action_values, source_tile_logits, target_tile_logits, state, env):
    Q = source_tile_logits.shape[1]
    R = source_tile_logits.shape[2]

    
    source_tile_logits_2d = source_tile_logits[0]  
    target_tile_logits_2d = target_tile_logits[0]
    action_values_2d = action_values[0]

    index_to_coord = {}
    for (q, r, s), hex_tile in env.game.atlas.landscape.items():
        grid_q = q + env.game.size
        grid_r = r + env.game.size
        idx = grid_q * R + grid_r
        index_to_coord[idx] = (q, r, s)

    unit_tensor = torch.tensor(state["grid"][1]).float().cuda()
    player_index = env.game.current_player_index
    player = env.game.players[player_index]

    valid_source_mask = torch.full((Q, R), float('-inf')).cuda()

    for (q, r, s), hex_tile in env.game.atlas.landscape.items():
        
        grid_q = q + env.game.size
        grid_r = r + env.game.size
        
        if unit_tensor[grid_q, grid_r] <= 0:
            continue

        source_hex = hex_tile
        potential_targets = env.game.atlas.neighbors_within_radius(source_hex, 2)
        has_valid_target = False
        for tgt in potential_targets:
            if env.game.can_we_do_that(player, source_hex, tgt, 'move/attack') or env.game.can_we_do_that(player, source_hex, tgt, 'build'):
                has_valid_target = True
                break
        
        if has_valid_target:
            valid_source_mask[grid_q, grid_r] = 0.0

    masked_source_logits = source_tile_logits_2d + valid_source_mask
    
    if torch.all(masked_source_logits == float('-inf')):
        return None
                    
    source_probs = torch.softmax(masked_source_logits.view(-1), dim=-1)
    source_dist = torch.distributions.Categorical(source_probs)
    
    source_idx = source_dist.sample()
    source_coords = index_to_coord[source_idx.cpu().item()]
    world_q, world_r, world_s = source_coords
    source_hex = env.game.atlas.get_hex(world_q, world_r, world_s)

    valid_target_mask = torch.full((Q, R), float('-inf')).cuda()
    possible_actions_for_target = {}

    neighbors_rad2 = env.game.atlas.neighbors_within_radius(source_hex, 2)
    for tgt in neighbors_rad2:
        can_0 = env.game.can_we_do_that(player, source_hex, tgt, 'move/attack')
        can_1 = env.game.can_we_do_that(player, source_hex, tgt, 'build')
        if can_0 or can_1:
            gq = tgt.q + env.game.size
            gr = tgt.r + env.game.size
            valid_target_mask[gq, gr] = 0.0
            valid_set = []
            if can_0:
                valid_set.append(0)
            if can_1:
                valid_set.append(1)
            possible_actions_for_target[(gq, gr)] = valid_set

    masked_target_logits = target_tile_logits_2d + valid_target_mask + env.mask.cuda()
    target_probs = torch.softmax(masked_target_logits.view(-1), dim=-1)
    target_dist = torch.distributions.Categorical(target_probs)
    
    target_idx = target_dist.sample()
    target_coords = index_to_coord[target_idx.cpu().item()]
    tw_q, tw_r, tw_s = target_coords
    
    t_q = tw_q + env.game.size
    t_r = tw_r + env.game.size

    valid_actions = possible_actions_for_target.get((t_q, t_r), [])
    chosen_action_value = action_values_2d[t_q, t_r]
    action_prob = torch.sigmoid(chosen_action_value)
    action_type_dist = torch.distributions.Bernoulli(action_prob)
    
    if len(valid_actions) == 2:
        action_type = action_type_dist.sample().long()
        
    elif len(valid_actions) == 1:
        action_type = torch.tensor(valid_actions[0]).cuda().long()
        
    else:
        raise ValueError("No valid actions for target!")


    
    return {
        'action_type': action_type,
        'action_type_distribution': action_type_dist,  
        'source_tile_idx': source_idx,
        'source_tile_distribution': source_dist,
        'target_tile_idx': target_idx,
        'target_tile_distribution': target_dist,
        'coordinates': {
            'source_q': world_q,
            'source_r': world_r,
            'target_q': tw_q,
            'target_r': tw_r
        }
    }








def main():
    epochs = 1000
    learning_rate = 0.001
    discount_factor = 0.98 #gamma
    trace_decay_rate = 0.9 #lambda
    initialized = True #whether we're not training from scratch
    size = 4

    env = CustomGameEnv(size)
    
    
    model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()
    target_model = ActorCriticNetwork((2, env.size, env.size), 2).cuda()
    if initialized:
        saved_state_dict = torch.load(f"size-{size}-actor_critic_model.pth")
        model.load_state_dict(saved_state_dict)
    
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
        
        target_model.load_state_dict(model.state_dict())
        while not done:
            timestep += 1
            grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).cuda()
            gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).cuda()
            
            action_values, source_tile_logits, target_tile_logits, value = model(grid_tensor, gold_tensor)
            
            sampling_results = sample_apply_masks(action_values, source_tile_logits, target_tile_logits, state, env)
            
            if sampling_results is None: 
                env.game.next_turn()
                state = env._get_observation()
                continue
            
            
            action_type = sampling_results['action_type']
            source_tile_idx = sampling_results['source_tile_idx']
            target_tile_idx = sampling_results['target_tile_idx']
            coords = sampling_results['coordinates']
            
            player = env.game.current_player_index
            
            next_state_this_pov, next_state_next_pov, reward, done, _ = env.step(                               #No good!
                (action_type, coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r']))
            
            
            episode_reward += reward
            

            
            if reward < 0: #invalid count is unused, if an invalid action is taken, the game breaks as it should be masked out.
                invalid += 1
            
            next_grid_tensor = torch.tensor(next_state_this_pov["grid"]).float().unsqueeze(0).cuda()
            next_gold_tensor = torch.tensor(next_state_this_pov["gold"]).float().unsqueeze(0).cuda()
            
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
                    if timestep % 2 == 0:
                        eligibility_traces_player1[name] = (trace_decay_rate * eligibility_traces_player1[name] + param.grad)
                        param.grad = eligibility_traces_player1[name] #* advantage.item() This explodes values. Idk why. I must be misinterpreting Sutton's book. 
                                                                                                                          #EDIT: Adam may be the problem?
                        #param.data += learning_rate * eligibility_traces_player1[name] * advantage.item()
                        #param.data -= learning_rate * 0.001 * param.data

                    else:
                        eligibility_traces_player2[name] = (trace_decay_rate * eligibility_traces_player2[name] + param.grad)
                        param.grad = eligibility_traces_player2[name] #* advantage.item()
                        #param.data += learning_rate * eligibility_traces_player2[name] * advantage.item()
                        #param.data -= learning_rate * 0.001 * param.data
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            

            
            total_optimizer.step()
            
            tau = 0.01
            for target_param, local_param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                
            
            if done or reward>0 or timestep % 100 == 0:
                grid_0 = state["grid"][0]
                grid_1 = state["grid"][1]
    
                grid_str = '\n'.join([' '.join(['🟦' if grid_0[i][j] == 0 and
                                                cell == 0 else '🟩' if grid_0[i][j] == 1 and 
                                                cell == 0 else '⬛' if grid_0[i][j] == -1 and 
                                                cell == 0 else str(cell) for j, cell in enumerate(row)]) for i, row in enumerate(grid_1)])                
                os.system('cls')
                sys.stdout.write(str(state["gold"]) + "\n")
                sys.stdout.flush()
                sys.stdout.write(grid_str)
                sys.stdout.flush()
                sys.stdout.write(f"\n Step: {timestep}, player: {player}, reward: {reward}, Epoch: {epoch}, " + 
                               f"action: {(action_type.item(), coords['source_q'] + env.game.size, coords['source_r'] + env.game.size, coords['target_q'] + env.game.size, coords['target_r'] + env.game.size)}\n" +
                               f"victories: {victories}, value, next_value: {value.item(), next_value.item()} \n" +
                               f"avg_reward_10_epoch: {sum(cumulative_rewards) / len(cumulative_rewards)}")
                sys.stdout.flush()
            
            state = next_state_next_pov
            
            if done:
                victories += 1
            elif timestep > 1000:
                done = True
        
        cumulative_rewards.append(episode_reward/timestep)
        rewards_per_timestep.append(episode_reward/timestep)

    torch.save(model.state_dict(), f"size-{size}-actor_critic_model.pth")
    print("Training complete!!")

    rewards_file = f"size-{size}-rewards.json"
    
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
    plt.savefig(f"./plots/size-{size}-aggregated_rewards.png")
    plt.show()
    
    
    
if __name__ == "__main__":
    main()