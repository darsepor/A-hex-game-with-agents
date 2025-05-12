import gym
from gym import spaces
import numpy as np
from game.game_logic import GameLogic
from game.player import ReinforcementAITraining
from game.entity import BattleShip, City, Soldier, Entity
import torch

# Vocabulary for tile entities
EMPTY_LAND = 0
EMPTY_WATER = 1
P1_SOLDIER = 2
P1_BATTLESHIP = 3
P1_CITY = 4
P2_SOLDIER = 5
P2_BATTLESHIP = 6
P2_CITY = 7
OUT_OF_BOUNDS = 8
VOCAB_SIZE = 9

class CustomGameEnv(gym.Env):
    def __init__(self, size, players = [ReinforcementAITraining, ReinforcementAITraining]):
        super(CustomGameEnv, self).__init__()
        self.game = GameLogic(players = players, size=size)

        #Action space: action_type(move/attack or build), source_q, source_r, target_q, target_r
        
        Q = self.game.size*2 +1
        R = self.game.size*2 +1
        #print(self.game.size)
        self.size = Q
        '''
        self.action_space = spaces.MultiDiscrete([2, Q, R, Q, R])
        self.observation_space = spaces.Dict({
    "grid": spaces.Box(
        low=np.full((2, Q, R), 0, dtype=np.int32),
        high=np.array([
            np.full((Q, R), 1, dtype=np.int32),
            np.full((Q, R), 6, dtype=np.int32)
        ], dtype=np.int32),
        shape=(2, Q, R),
        dtype=np.int32
    ),
    
    "gold": spaces.Box(
        low=0, high=10000,
        shape=(2,),
        dtype=np.int32
    )})
    '''
        self.mask = torch.full((Q, R), float('-inf'), dtype=torch.float32)

        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size

            self.mask[grid_q, grid_r] = 0.0
        
        
    def reset(self, new_size):
        self.game = GameLogic(size=new_size, players=[ReinforcementAITraining, ReinforcementAITraining])
        self.mask = torch.full((new_size*2+1, new_size*2+1), float('-inf'), dtype=torch.float32)
        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size

            self.mask[grid_q, grid_r] = 0.0
        initial_observation = self._get_observation()
        return initial_observation
    
    def step(self, action):
        done = False
        
        action_type, source_q, source_r, target_q, target_r = action

        source_tile = self.game.atlas.get_hex(source_q, source_r, -source_q - source_r)
        target_tile = self.game.atlas.get_hex(target_q, target_r, -target_q - target_r)
        success = False
        reward = 0
        if action_type == 0:  # Move/Attack
      
            if target_tile.unit is not None:
                reward +=0.1
                if isinstance(target_tile.unit, City):
                    reward += 1
                self.game.attack_unit(source_tile.unit, target_tile.unit)
                
            else:
                self.game.move_unit(source_tile.unit, source_tile, target_tile)
            

        elif action_type == 1:  # Build
             if isinstance(source_tile.unit, City):
                    if target_tile.is_water:
                        self.game.place_battleship(source_tile.unit.owner,source_tile, target_tile)
                        
                    else:
                        self.game.place_soldier(source_tile.unit.owner, source_tile, target_tile)

                        
                    
             else:
                self.game.build_city(source_tile.unit.owner, source_tile, target_tile)
        #player_index = self.game.current_player_index
        #current_player = self.game.players[player_index]
        #opponent = self.game.players[(player_index + 1) % 2]
        #reward += 0.1 * (len(current_player.units) - len(opponent.units))

        

                
        if self.game.game_over:
            reward +=500
            done = True

        obs_this_pov = self._get_observation()
        
        self.game.next_turn()
            
        obs_next_pov = self._get_observation()
        
        return obs_this_pov, obs_next_pov, reward, done, {}
    
    def _get_observation(self):
        Q = self.game.size * 2 + 1
        R = self.game.size * 2 + 1
        # Create ID grid for entities and terrain
        id_grid = np.full((Q, R), OUT_OF_BOUNDS, dtype=np.int64)
        player = self.game.current_player_index
        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size
            if hex_tile.unit is None:
                # Empty tile: land or water
                id_grid[grid_q, grid_r] = EMPTY_WATER if hex_tile.is_water else EMPTY_LAND
            else:
                # Unit present: soldier, battleship, or city
                unit = hex_tile.unit
                if isinstance(unit, Soldier):
                    id_grid[grid_q, grid_r] = P1_SOLDIER if unit.owner == self.game.players[player] else P2_SOLDIER
                elif isinstance(unit, BattleShip):
                    id_grid[grid_q, grid_r] = P1_BATTLESHIP if unit.owner == self.game.players[player] else P2_BATTLESHIP
                elif isinstance(unit, City):
                    id_grid[grid_q, grid_r] = P1_CITY if unit.owner == self.game.players[player] else P2_CITY
        # Gold values for current player and opponent
        gold_values = np.array([self.game.players[player].currency,
                                 self.game.players[(player + 1) % 2].currency],
                                dtype=np.int32)
        observation = {
            "grid": id_grid,  # (Q, R) integer IDs
            "gold": gold_values
        }
        return observation
            
    
    def sample_apply_masks(self, action_values, source_tile_logits, target_tile_logits, state, device,
                           force_source_idx=None, force_target_idx=None, force_action_type=None):
        """Applies masks, samples or uses forced indices, and returns chosen action + distributions.
        
        If force_* args are None, samples normally.
        If force_* args are provided, uses them instead of sampling but still calculates distributions
        based on the provided model outputs (logits/values) and the derived masks.
        
        Returns a dictionary containing chosen indices, coordinates, distributions, masks, and dict.
        """
        Q = source_tile_logits.shape[1]
        R = source_tile_logits.shape[2]

        action_values_2d = action_values[0]
        source_tile_logits_2d = source_tile_logits[0]  
        target_tile_logits_2d = target_tile_logits[0]

        index_to_coord = {} # Map flat index to (q, r, s) world coords
        # coord_to_index = {} # Map (q, r, s) world coords to flat index (unused currently but potentially useful)
        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size
            idx = grid_q * R + grid_r
            index_to_coord[idx] = (q, r, s)
            # coord_to_index[(q,r,s)] = idx

        unit_tensor = torch.tensor(state["grid"]).long().to(device)
        player_index = self.game.current_player_index
        player = self.game.players[player_index]

        valid_source_mask = torch.full((Q, R), float('-inf')).to(device)
        for (q, r, s), hex_tile_from_atlas in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size
            if unit_tensor[grid_q, grid_r] < P1_SOLDIER:
                continue

            potential_targets = self.game.atlas.neighbors_within_radius(hex_tile_from_atlas, 2)
            has_valid_target = False
            for tgt in potential_targets:
                if self.game.can_we_do_that(player, hex_tile_from_atlas, tgt, 'move/attack') or \
                   self.game.can_we_do_that(player, hex_tile_from_atlas, tgt, 'build'):
                    has_valid_target = True
                    break
            if has_valid_target:
                valid_source_mask[grid_q, grid_r] = 0.0

        masked_source_logits = source_tile_logits_2d + valid_source_mask
        source_probs = torch.softmax(masked_source_logits.view(-1), dim=-1)
        source_dist = torch.distributions.Categorical(source_probs)

        if force_source_idx is not None:
            source_idx = force_source_idx.long()
        else:
            if torch.all(masked_source_logits == float('-inf')):
                 return None
            source_idx = source_dist.sample()

        source_coords = index_to_coord[source_idx.cpu().item()]
        world_q, world_r, world_s = source_coords
        source_hex = self.game.atlas.get_hex(world_q, world_r, world_s)
        
        # --- Target Selection (depends on chosen source) --- 
        valid_target_mask = torch.full((Q, R), float('-inf')).to(device)
        possible_actions_for_target = {}
        neighbors_rad2 = self.game.atlas.neighbors_within_radius(source_hex, 2)
        for tgt in neighbors_rad2:
            can_0 = self.game.can_we_do_that(player, source_hex, tgt, 'move/attack')
            can_1 = self.game.can_we_do_that(player, source_hex, tgt, 'build')
            if can_0 or can_1:
                gq = tgt.q + self.game.size
                gr = tgt.r + self.game.size
                valid_target_mask[gq, gr] = 0.0
                valid_set = []
                if can_0: valid_set.append(0)
                if can_1: valid_set.append(1)
                possible_actions_for_target[(gq, gr)] = valid_set

        combined_target_mask = valid_target_mask + self.mask.to(device)
        masked_target_logits = target_tile_logits_2d + combined_target_mask
        target_probs = torch.softmax(masked_target_logits.view(-1), dim=-1)
        target_dist = torch.distributions.Categorical(target_probs)
        
        if force_target_idx is not None:
            target_idx = force_target_idx.long()
        else:
             if torch.all(masked_target_logits == float('-inf')):
                 # Chosen source has no valid targets for the current target_tile_logits.
                 return None # If sampling, indicate failure for this path.
             target_idx = target_dist.sample()

        target_coords_world = index_to_coord[target_idx.cpu().item()]
        tw_q, tw_r, tw_s = target_coords_world
        
        # --- Action Type Selection (depends on chosen target) --- 
        target_grid_q = tw_q + self.game.size
        target_grid_r = tw_r + self.game.size
        valid_actions = possible_actions_for_target.get((target_grid_q, target_grid_r), [])
        
        action_type = None
        action_type_dist = None # Stores Bernoulli distribution (or a deterministic substitute)
        
        if len(valid_actions) == 2: # Both move/attack and build are possible for source-target pair
            chosen_action_value = action_values_2d[target_grid_q, target_grid_r]
            action_prob = torch.sigmoid(chosen_action_value) # Prob of choosing action type 1 (build)
            action_type_dist = torch.distributions.Bernoulli(action_prob)
            if force_action_type is not None:
                action_type = force_action_type.long()
            else:
                action_type = action_type_dist.sample().long()
        elif len(valid_actions) == 1: # Only one action type is possible
            action_type = torch.tensor(valid_actions[0]).to(device).long()
            # Create a Bernoulli dist that deterministically yields the only valid action for log_prob calculations.
            deterministic_prob = torch.tensor(1.0 if valid_actions[0] == 1 else 0.0, device=device)
            action_type_dist = torch.distributions.Bernoulli(deterministic_prob) 
            if force_action_type is not None and force_action_type.long().item() != action_type.item():
                 # Consistency check for forced mode.
                 print(f"Warning: Forced action type {force_action_type.item()} mismatches the only valid action {action_type.item()} for target ({tw_q},{tw_r}) from source ({world_q},{world_r}). Using forced action.")
                 action_type = force_action_type.long()
        else: # No valid actions for the chosen/forced source-target pair.
             if force_target_idx is None: # Problem if we sampled this target.
                  print(f"Warning: Sampled target ({tw_q},{tw_r}) has no valid actions from source ({world_q},{world_r}). Returning None.")
                  return None
             else: # Problem if target was forced, indicates bad buffer data or logic.
                 raise ValueError(f"Forced target ({tw_q},{tw_r}) has no valid actions from source ({world_q},{world_r}). Invalid state/action data.")

        if action_type is None: # Should be unreachable if logic above is sound.
            raise ValueError("Action type selection failed unexpectedly.")
        
        return {
            'action_type': action_type,
            'source_tile_idx': source_idx,
            'target_tile_idx': target_idx,
            'action_type_distribution': action_type_dist,
            'source_tile_distribution': source_dist,
            'target_tile_distribution': target_dist,
            'coordinates': {
                'source_q': world_q,
                'source_r': world_r,
                'target_q': tw_q,
                'target_r': tw_r
            },
            'valid_source_mask': valid_source_mask,
            'valid_target_mask': valid_target_mask,
            'possible_actions_for_target': possible_actions_for_target
        }
