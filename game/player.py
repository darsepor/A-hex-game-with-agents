from abc import ABC, abstractmethod
import random
from game.entity import Entity, Soldier, City, BattleShip
import torch
import numpy as np
from res_net_AC import ResActorCriticNetwork
import os
import sys

class Player(ABC):
    def __init__(self, name, color):
        self.name = name
        self.currency = 50
        self.units = []
        self.cities = []
        self.color = color

    def adjust_currency(self, amount):
        self.currency += amount

    @abstractmethod
    def take_turn(self, game_logic):
        pass

    @property
    def is_ai(self):
        return False  # Default to False for human players


class Human(Player):
    def __init__(self, name, color):
        super().__init__(name, color)

    def take_turn(self, game_logic):
        # Human player's turn is managed via GUI events
        pass

    @property
    def is_ai(self):
        return False


class SimpleAI(Player): #doesn't really work well, for testing the game
    def __init__(self, name, color):
        super().__init__(name, color)

    def take_turn(self, game_logic):
        if game_logic.game_over:
            return

        possessions = [hex_tile for hex_tile in game_logic.atlas.landscape.values()
                       if hex_tile.unit is not None and hex_tile.unit.owner == self]

        #Categorize possessions into soldiers, ships, and cities
        soldiers, ships, cities = self.soldiers_ships_cities(possessions)

        #Attempt to attack first
        attacked = self.attack_target_first_ai(soldiers, ships, game_logic)
        if attacked:
            return

        opponent = next(player for player in game_logic.players if player != self)

        first_unit_human = next((hex_tile for hex_tile in game_logic.atlas.landscape.values()
                                 if hex_tile.unit is not None and hex_tile.unit.owner == opponent),
                                None)
        

        if self.currency >= game_logic.dynamic_city_cost(self):
            chance = random.random()
            if chance <= 0.3 and self.currency >= Entity.soldier_cost:
                soldier_candidates = []
                for city_hex in cities:
                    neighbors = game_logic.atlas.neighbors(city_hex)
                    for neighbor in neighbors:
                        if neighbor.unit is None and not neighbor.is_water:
                            soldier_candidates.append((city_hex, neighbor))
                if soldier_candidates:
                    source, target = min(soldier_candidates,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile[1], first_unit_human))
                    game_logic.place_soldier(self, source, target)
                    return
            if chance <= 0.6 and self.currency >= Entity.ship_cost:
                ship_candidates = []
                for city_hex in cities:
                    neighbors = game_logic.atlas.neighbors(city_hex)
                    for neighbor in neighbors:
                        if neighbor.unit is None and neighbor.is_water:
                            ship_candidates.append((city_hex, neighbor))
                if ship_candidates:
                    source, target = min(ship_candidates,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile[1], first_unit_human))
                    game_logic.place_battleship(self, source, target)
                    return
                
            city_candidates = []
            for possession in soldiers.union(ships):
                neighbors = game_logic.atlas.neighbors(possession)
                for neighbor in neighbors:
                    if neighbor.unit is None and not neighbor.is_water:
                        city_candidates.append((possession, neighbor))
            if city_candidates:
                source, target = min(city_candidates,
                                key=lambda hex_tile: game_logic.atlas.distance(hex_tile[1], first_unit_human))
                game_logic.build_city(self, source, target)
                return
        else:
            human_possessions = [hex_tile for hex_tile in game_logic.atlas.landscape.values()
                                 if hex_tile.unit is not None and hex_tile.unit.owner == opponent]
            for movable_hex in soldiers.union(ships):
                unit = movable_hex.unit
                if isinstance(unit, City):
                    continue
                for target_hex in human_possessions:                          #min(human_possessions,
                                                                              #key=lambda hex_tile: game_logic.atlas.distance(hex_tile, movable_hex))
                    if target_hex is not None:
                        success = self.move_unit_towards_target(unit, movable_hex, target_hex, game_logic)
                        if success:
                            return

    def soldiers_ships_cities(self, hex_tiles):
        soldiers = set()
        ships = set()
        cities = set()
        for hex_tile in hex_tiles:
            unit = hex_tile.unit
            if isinstance(unit, Soldier):
                soldiers.add(hex_tile)
            elif isinstance(unit, BattleShip):
                ships.add(hex_tile)
            elif isinstance(unit, City):
                cities.add(hex_tile)
        return soldiers, ships, cities

    def attack_target_first_ai(self, soldiers, ships, game_logic):
        opponent = next(player for player in game_logic.players if player != self)

        soldier_targets = set()
        for soldier_hex in soldiers:
            neighbors = game_logic.atlas.neighbors(soldier_hex)
            for neighbor in neighbors:
                if neighbor.unit is not None and neighbor.unit.owner == opponent:
                    soldier_targets.add((neighbor, 'soldier', soldier_hex))

        ship_targets = set()
        for ship_hex in ships:
            neighbors = game_logic.atlas.neighbors_within_radius(ship_hex, 2)
            for neighbor in neighbors:
                if neighbor.unit is not None and neighbor.unit.owner == opponent:
                    ship_targets.add((neighbor, 'ship', ship_hex))

        targets = []
        for target_set in [soldier_targets, ship_targets]:
            for target_hex, attack_type, attacker_hex in target_set:
                unit = target_hex.unit
                if isinstance(unit, BattleShip):
                    priority = 2
                elif isinstance(unit, Soldier):
                    priority = 1
                elif isinstance(unit, City):
                    priority = 3
                else:
                    continue
                targets.append((priority, target_hex, attack_type, attacker_hex))

        if targets:
            targets.sort(key=lambda x: x[0])
            # Random chance to select the best target or a random one
            rand_double = random.random()
            if rand_double <= 0.8:
                final_target = targets[0]
            else:
                final_target = random.choice(targets)
            _, target_hex, attack_type, attacker_hex = final_target
            attacker = attacker_hex.unit
            if attacker is not None and target_hex.unit is not None:
                game_logic.attack_unit(attacker, target_hex.unit)

                return True
        return False

    def move_unit_towards_target(self, unit, unit_hex, target_hex, game_logic):
        path = game_logic.find_path(unit_hex, target_hex, unit)
        if len(path) > 1:
            next_hex = path[1]
            if next_hex.unit is None and (not next_hex.is_water if isinstance(unit, Soldier) else next_hex.is_water):
                game_logic.move_unit(unit, unit_hex, next_hex)
                return True
        return False

    @property
    def is_ai(self):
        return True




class ReinforcementAITraining(Player):
    def __init__(self, name, color):
        super().__init__(name, color)

    def take_turn(self, game_logic):

        pass #This is just for training. Should make a class that loads saved model files.

    @property
    def is_ai(self):
        return True
    
class ANNAI(Player): #This is for evaluation
    def __init__(self, name, color, size=4, device='cuda'):
        super().__init__(name, color)
        self.device = device
        self.model = ResActorCriticNetwork((2, size*2+1, size*2+1), 2).to(self.device)
        self.model.load_state_dict(torch.load("overall_curriculum_actor_critic_model.pth", map_location=self.device))
        self.model.eval()
        
        
    def take_turn(self, game_logic):
        if game_logic.game_over:
            return
        
        state = get_observation(game_logic)
        grid_tensor = torch.tensor(state["grid"]).float().unsqueeze(0).to(self.device)
        gold_tensor = torch.tensor(state["gold"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, source_tile_logits, target_tile_logits, _ = self.model(grid_tensor, gold_tensor)
        sampling_results = greedy_apply_masks(game_logic, action_values, source_tile_logits, target_tile_logits, state, self.device)
        if sampling_results is None:
            return
        action_type = sampling_results['action_type'].item()
        coords = sampling_results['coordinates']
        neural_network_step((action_type, coords['source_q'], coords['source_r'], coords['target_q'], coords['target_r']), game_logic)

        
        
        
        
        
        
        
#This isn't good but the environment class was build pretty much for training only and I'm not sure how to unentangle it all. Furthermore, we want to do greedy sampling.
def get_observation(game):
    Q = game.size * 2 + 1
    R = game.size * 2 + 1
    terrain_layer = np.full((Q, R), -1, dtype=np.int32)
                                                           
    units_layer = np.full((Q, R), 0, dtype=np.int32)
    player = game.current_player_index
    for (q, r, s), hex_tile in game.atlas.landscape.items():
        grid_q = q + game.size
        grid_r = r + game.size

        if hex_tile.is_water:
            terrain_layer[grid_q, grid_r] = 0
        else:
            terrain_layer[grid_q, grid_r] = 1

        if hex_tile.unit is None:
            units_layer[grid_q, grid_r] = 0
        else:
            unit = hex_tile.unit
            if isinstance(unit, Soldier):
                units_layer[grid_q, grid_r] = 3 if unit.owner == game.players[player] else -3
            elif isinstance(unit, BattleShip):
                units_layer[grid_q, grid_r] = 7 if unit.owner == game.players[player] else -7
            elif isinstance(unit, City):
                units_layer[grid_q, grid_r] = 20 if unit.owner == game.players[player] else -20
    
    gold_values = np.array([game.players[player].currency, game.players[(player + 1) % 2].currency], dtype=np.int32)

    observation = {
        "grid": np.array([terrain_layer, units_layer], dtype=np.int32),
        "gold": gold_values
    }
    grid_0 = observation["grid"][0]
    grid_1 = observation["grid"][1]

    grid_str = '\n'.join([' '.join(['🟦' if grid_0[i][j] == 0 and
                                    cell == 0 else '🟩' if grid_0[i][j] == 1 and 
                                    cell == 0 else '⬛' if grid_0[i][j] == -1 and 
                                    cell == 0 else str(cell) for j, cell in enumerate(row)]) for i, row in enumerate(grid_1)])                
    os.system('cls')
    sys.stdout.write(str(observation["gold"]) + "\n")
    sys.stdout.flush()
    sys.stdout.write(grid_str)
    sys.stdout.flush()
    
    
    
    return observation
        



def greedy_apply_masks(game, action_values, source_tile_logits, target_tile_logits, state, device):
    Q = source_tile_logits.shape[1]
    R = source_tile_logits.shape[2]

    
    source_tile_logits_2d = source_tile_logits[0]  
    target_tile_logits_2d = target_tile_logits[0]
    action_values_2d = action_values[0]

    index_to_coord = {}
    for (q, r, s), hex_tile in game.atlas.landscape.items():
        grid_q = q + game.size
        grid_r = r + game.size
        idx = grid_q * R + grid_r
        index_to_coord[idx] = (q, r, s)

    unit_tensor = torch.tensor(state["grid"][1]).float().to(device)
    player_index = game.current_player_index
    player = game.players[player_index]

    valid_source_mask = torch.full((Q, R), float('-inf')).to(device)

    for (q, r, s), hex_tile in game.atlas.landscape.items():
        
        grid_q = q + game.size
        grid_r = r + game.size
        
        if unit_tensor[grid_q, grid_r] <= 0:
            continue

        source_hex = hex_tile
        potential_targets = game.atlas.neighbors_within_radius(source_hex, 2)
        has_valid_target = False
        for tgt in potential_targets:
            if game.can_we_do_that(player, source_hex, tgt, 'move/attack') or game.can_we_do_that(player, source_hex, tgt, 'build'):
                has_valid_target = True
                break
        
        if has_valid_target:
            valid_source_mask[grid_q, grid_r] = 0.0

    masked_source_logits = source_tile_logits_2d + valid_source_mask
    
    if torch.all(masked_source_logits == float('-inf')):
        return None
                    
    masked_source_1d = masked_source_logits.view(-1)
    source_idx = masked_source_1d.argmax(dim=0)
    source_coords = index_to_coord[source_idx.cpu().item()]
    world_q, world_r, world_s = source_coords
    source_hex = game.atlas.get_hex(world_q, world_r, world_s)

    valid_target_mask = torch.full((Q, R), float('-inf')).to(device)
    possible_actions_for_target = {}

    neighbors_rad2 = game.atlas.neighbors_within_radius(source_hex, 2)
    for tgt in neighbors_rad2:
        can_0 = game.can_we_do_that(player, source_hex, tgt, 'move/attack')
        can_1 = game.can_we_do_that(player, source_hex, tgt, 'build')
        if can_0 or can_1:
            gq = tgt.q + game.size
            gr = tgt.r + game.size
            valid_target_mask[gq, gr] = 0.0
            valid_set = []
            if can_0:
                valid_set.append(0)
            if can_1:
                valid_set.append(1)
            possible_actions_for_target[(gq, gr)] = valid_set

    masked_target_logits = target_tile_logits_2d + valid_target_mask
    
    masked_target_1d = masked_target_logits.view(-1)
    target_idx = masked_target_1d.argmax(dim=0)
    target_coords = index_to_coord[target_idx.cpu().item()]
    tw_q, tw_r, tw_s = target_coords
    
    t_q = tw_q + game.size
    t_r = tw_r + game.size

    valid_actions = possible_actions_for_target.get((t_q, t_r), [])
    chosen_action_value = action_values_2d[t_q, t_r]
    action_prob = torch.sigmoid(chosen_action_value)
    
    if len(valid_actions) == 2:
        # choose action 0 if prob < 0.5 else action 1
        if action_prob.item() < 0.5:
            action_type = torch.tensor(valid_actions[0]).long().to(device)
        else:
            action_type = torch.tensor(valid_actions[1]).long().to(device)
    elif len(valid_actions) == 1:
        action_type = torch.tensor(valid_actions[0]).long().to(device)
    else:
        raise ValueError("No valid actions for target!")


    
    return {
        'action_type': action_type,
        'coordinates': {
            'source_q': world_q,
            'source_r': world_r,
            'target_q': tw_q,
            'target_r': tw_r
        }
    }
    
def neural_network_step(action, game):
        
        action_type, source_q, source_r, target_q, target_r = action

        source_tile = game.atlas.get_hex(source_q, source_r, -source_q - source_r)
        target_tile = game.atlas.get_hex(target_q, target_r, -target_q - target_r)
        if action_type == 0:  # Move/Attack
      
            if target_tile.unit is not None:

                game.attack_unit(source_tile.unit, target_tile.unit)
                
            else:
                game.move_unit(source_tile.unit, source_tile, target_tile)
            

        elif action_type == 1:  # Build
             if isinstance(source_tile.unit, City):
                    if target_tile.is_water:
                        game.place_battleship(source_tile.unit.owner,source_tile, target_tile)
                        
                    else:
                        game.place_soldier(source_tile.unit.owner, source_tile, target_tile)

                        
                    
             else:
                game.build_city(source_tile.unit.owner, source_tile, target_tile)