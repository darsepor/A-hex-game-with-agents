import gym
from gym import spaces
import numpy as np
from game.game_logic import GameLogic
from game.player import ReinforcementAI
from game.entity import BattleShip, City, Soldier, Entity
import torch

class CustomGameEnv(gym.Env):
    def __init__(self, size, mode='reinforcementai_vs_reinforcementai'):
        super(CustomGameEnv, self).__init__()
        self.game = GameLogic(mode=mode, size=size)

        #Action space: action_type(move/attack or build), source_q, source_r, target_q, target_r
        
        Q = self.game.size*2 +1
        R = self.game.size*2 +1
        #print(self.game.size)
        self.size = Q
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
        self.mask = torch.full((Q, R), float('-inf'), dtype=torch.float32)

        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size

            self.mask[grid_q, grid_r] = 0.0
        
        
    def reset(self, new_size):
        self.game = GameLogic(size=new_size, mode='reinforcementai_vs_reinforcementai')
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
            attacking = target_tile is not None and target_tile.unit is not None and target_tile.unit.owner != self.game.players[self.game.current_player_index] and isinstance(target_tile.unit, City)
                
            success = self._handle_move_attack(source_tile, target_tile)
            if attacking and success:
                reward = 1

        elif action_type == 1:  # Build
            success = self._handle_build(source_tile, target_tile)
        #print(source_tile.unit == True)
        #print(source_tile.unit.owner == self.game.current_player_index == True)
        
        if not success:
            dist =  abs(source_q - target_q) + abs(source_r - target_r)
            reward = -1
            #if dist > 5:
            #    reward = -20
            #print(source_tile.unit == True)
            #print(source_tile.unit.owner == self.game.current_player_index == True)
            #if source_tile.unit and source_tile.unit.owner == self.game.current_player_index:
            #    reward = 10
                
        elif self.game.game_over:
            reward +=100
            done = True
        
        #print(self.game.current_player_index)
        #if success:
        obs_this_pov = self._get_observation()
        
        self.game.next_turn()
            
        #print(f"gugsdfg {self.game.current_player_index}")
        obs_next_pov = self._get_observation()
        
        return obs_this_pov, obs_next_pov, reward, done, {}
    
    def _get_observation(self):
        Q = self.game.size * 2 + 1
        R = self.game.size * 2 + 1
        terrain_layer = np.full((Q, R), -1, dtype=np.int32)  # Initialize with -1 for unused areas,
                                                             #as it is an injective map from hex grid to observation space
        units_layer = np.full((Q, R), 0, dtype=np.int32)     #Same for units EDIT: better to just apply a mask to logits i guess
        player = self.game.current_player_index
        for (q, r, s), hex_tile in self.game.atlas.landscape.items():
            grid_q = q + self.game.size
            grid_r = r + self.game.size

            if hex_tile.is_water:
                terrain_layer[grid_q, grid_r] = 0
            else:
                terrain_layer[grid_q, grid_r] = 1

            if hex_tile.unit is None:
                units_layer[grid_q, grid_r] = 0
            else:
                unit = hex_tile.unit
                if isinstance(unit, Soldier):
                    units_layer[grid_q, grid_r] = 3 if unit.owner == self.game.players[player] else -3
                elif isinstance(unit, BattleShip):
                    units_layer[grid_q, grid_r] = 7 if unit.owner == self.game.players[player] else -7
                elif isinstance(unit, City):
                    units_layer[grid_q, grid_r] = 20 if unit.owner == self.game.players[player] else -20
        
        gold_values = np.array([self.game.players[player].currency, self.game.players[(player + 1) % 2].currency], dtype=np.int32)

        observation = {
            "grid": np.array([terrain_layer, units_layer], dtype=np.int32),
            "gold": gold_values
        }

        return observation
            
    
    
    def _handle_move_attack(self, source_tile, target_tile):
        if source_tile is None or target_tile is None or source_tile.unit is None or isinstance(source_tile.unit, City):
            return False

        unit = source_tile.unit
        if(unit.owner != self.game.players[self.game.current_player_index]):
            return False
        
        if isinstance(unit, Soldier) and target_tile not in self.game.atlas.neighbors(source_tile):
            return False        
        
        if isinstance(unit, BattleShip) and target_tile not in self.game.atlas.neighbors_within_radius(source_tile, 2):
            return False  
        
        
        if target_tile.unit is not None and target_tile.unit.owner != unit.owner:
            self.game.attack_unit(unit, target_tile.unit)
            return True

        if target_tile.unit is not None:
            return False
        
        
        if isinstance(unit, Soldier) and not target_tile.is_water:
            
            self.game.move_unit(unit, source_tile, target_tile)
            return True
            
        if isinstance(unit, BattleShip) and target_tile.is_water:
        
            radius_one = self.game.atlas.neighbors_within_radius(source_tile, 1)
            radius_two = []
            for tile in radius_one:
                if tile.is_water and tile.unit is None:
                    neighbors = self.game.atlas.neighbors_within_radius(tile, 1)
                    radius_two.extend([t for t in neighbors if t.is_water and t.unit is None])
            highlighted_tiles = [tile for tile in set(radius_one + radius_two) if tile.is_water and tile.unit is None]
            
            if(target_tile in highlighted_tiles):
                self.game.move_unit(unit, source_tile, target_tile)
                return True

        
        return False

    def _handle_build(self, source_tile, target_tile):
        if source_tile is None or target_tile is None or source_tile.unit is None:
            return False
        
        
        if target_tile not in self.game.atlas.neighbors(source_tile):
            return False
        
        unit = source_tile.unit
        if unit.owner != self.game.players[self.game.current_player_index]:
            return False
        
        
        if isinstance(unit, City):
            if target_tile.unit is None:
                if target_tile.is_water and unit.owner.currency >= Entity.ship_cost:
                    self.game.place_battleship(source_tile.unit.owner, target_tile)
                    unit.owner.adjust_currency(-Entity.ship_cost)
                    return True
                elif not target_tile.is_water and unit.owner.currency >= Entity.soldier_cost:
                    self.game.place_soldier(source_tile.unit.owner, target_tile)
                    unit.owner.adjust_currency(-Entity.soldier_cost)
                    return True
                
        else:
            if target_tile.unit is None and not target_tile.is_water and unit.owner.currency>= Entity.city_cost* (1.6**(len(unit.owner.cities)-1)):
                self.game.build_city(source_tile.unit.owner, target_tile)
                unit.owner.adjust_currency(-Entity.city_cost* (1.6**(len(unit.owner.cities)-1)))
                return True

            return False


