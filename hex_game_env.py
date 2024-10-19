import gym
from gym import spaces
import numpy as np
from game.game_logic import GameLogic
from game.player import ReinforcementAI
from game.entity import BattleShip, City, Soldier, Entity

class CustomGameEnv(gym.Env):
    def __init__(self, mode='reinforcementai_vs_reinforcementai'):
        super(CustomGameEnv, self).__init__()
        self.game = GameLogic(mode=mode)

        #Action space: [action_type(move/attack or build), source_q, source_r, target_q, target_r]
        
        Q = self.game.size*2 +1
        R = self.game.size*2 +1
        self.action_space = spaces.MultiDiscrete([2, Q, R, Q, R])
        self.observation_space = spaces.Dict({
    "grid": spaces.Box(
        low=np.array([[0, 0]]),
        high=np.array([[1, 6]]),
        shape=(2, Q, R),
        dtype=np.int32),
    
    "gold": spaces.Box(
        low=0, high=10000,
        shape=(2,),
        dtype=np.int32
    )
})
        
    def step(self, action):
        done = False
        
        action_type, source_q, source_r, target_q, target_r = action

        source_tile = self.game.atlas.get_hex(source_q, source_r, -source_q - source_r)
        target_tile = self.game.atlas.get_hex(target_q, target_r, -target_q - target_r)
        success = False
        if action_type == 0:  # Move/Attack
            success = self._handle_move_attack(source_tile, target_tile)

        elif action_type == 1:  # Build
            success = self._handle_build(source_tile, target_tile)

        reward = 0
        if not success:
            reward += -10
        elif self.game.game_over:
            reward +=500
            done = True
        self.game.next_turn
        obs = self._get_observation()
        
        return obs, reward, done, {}
    
    
    #FIX ME: Abomination
    
    
    def _handle_move_attack(self, source_tile, target_tile):
        if not source_tile or not target_tile or not source_tile.unit or isinstance(source_tile.unit, City):
            return False

        unit = source_tile.unit
        if(unit.owner != self.game.players[self.game.current_player_index]):
            return False
        
        if isinstance(unit, Soldier) and target_tile not in self.game.atlas.neighbors(source_tile):
            return False        
        
        if isinstance(unit, BattleShip) and target_tile not in self.game.atlas.neighbors_within_radius(source_tile, 2):
            return False  
        
        
        if target_tile.unit and target_tile.unit.owner != unit.owner:
            if isinstance(unit, Soldier):
                target_tile.unit.attacked_by_ship
                return True
            else:
                target_tile.unit.attacked_by_soldier
                return True

        if target_tile.unit:
            return False
        
        
        if isinstance(unit, Soldier) and not target_tile.is_water:
            
            self.game.move_unit(unit, source_tile, target_tile)
            return True
            
        if isinstance(unit, BattleShip) and target_tile.is_water:
        
            radius_one = self.game_logic.atlas.neighbors_within_radius(source_tile, 1)
            radius_two = []
            for tile in radius_one:
                if tile.is_water and tile.unit is None:
                    neighbors = self.game_logic.atlas.neighbors_within_radius(tile, 1)
                    radius_two.extend([t for t in neighbors if t.is_water and t.unit is None])
            highlighted_tiles = [tile for tile in set(radius_one + radius_two) if tile.is_water and tile.unit is None]
            
            if(target_tile in highlighted_tiles):
                self.game.move_unit(unit, source_tile, target_tile)
                return True

        
        return False

    def _handle_build(self, source_tile, target_tile):
        if not source_tile or not target_tile or not source_tile.unit or not source_tile.unit.owner:
            return False
        unit = source_tile.unit
        if(unit.owner != self.game.players[self.game.current_player_index]):
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
            if target_tile.unit is None and not target_tile.is_water and unit.owner.currency>= Entity.city_cost:
                self.game.build_city(source_tile.unit.owner, target_tile)
                unit.owner.adjust_currency(-Entity.city_cost)
                return True

            return False

