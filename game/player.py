
from abc import ABC, abstractmethod
import random
from game.entity import Entity, Soldier, City, BattleShip

class Player(ABC):
    def __init__(self, name, color):
        self.name = name
        self.currency = 100
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


class SimpleAI(Player): #doesn't really work, for testing the game
    def __init__(self, name, color):
        super().__init__(name, color)

    def take_turn(self, game_logic):
        if game_logic.game_over:
            return

        possessions = [hex_tile for hex_tile in game_logic.atlas.landscape.values()
                       if hex_tile.unit and hex_tile.unit.owner == self]

        #Categorize possessions into soldiers, ships, and cities
        soldiers, ships, cities = self.soldiers_ships_cities(possessions)

        #Attempt to attack first
        attacked = self.attack_target_first_ai(soldiers, ships, game_logic)
        if attacked:
            return

        opponent = next(player for player in game_logic.players if player != self)

        first_unit_human = next((hex_tile for hex_tile in game_logic.atlas.landscape.values()
                                 if hex_tile.unit and hex_tile.unit.owner == opponent),
                                None)
        

        if self.currency >= Entity.city_cost and first_unit_human:
            chance = random.random()
            if chance <= 0.5 and self.currency >= Entity.soldier_cost:
                soldier_candidates = []
                for city_hex in cities:
                    neighbors = game_logic.atlas.neighbors(city_hex)
                    for neighbor in neighbors:
                        if neighbor.unit is None and not neighbor.is_water:
                            soldier_candidates.append(neighbor)
                if soldier_candidates:
                    location = min(soldier_candidates,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile, first_unit_human))
                    game_logic.place_soldier(self, location)
                    self.adjust_currency(-Entity.soldier_cost)
                    return
            elif chance <= 0.8 and self.currency >= Entity.ship_cost:
                ship_candidates = []
                for city_hex in cities:
                    neighbors = game_logic.atlas.neighbors(city_hex)
                    for neighbor in neighbors:
                        if neighbor.unit is None and neighbor.is_water:
                            ship_candidates.append(neighbor)
                if ship_candidates:
                    location = min(ship_candidates,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile, first_unit_human))
                    game_logic.place_battleship(self, location)
                    self.adjust_currency(-Entity.ship_cost)
                    return
            else:
                city_candidates = []
                for possession in possessions:
                    neighbors = game_logic.atlas.neighbors(possession)
                    for neighbor in neighbors:
                        if neighbor.unit is None and not neighbor.is_water:
                            city_candidates.append(neighbor)
                if city_candidates:
                    location = min(city_candidates,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile, first_unit_human))
                    game_logic.build_city(self, location)
                    self.adjust_currency(-Entity.city_cost)
                    return
        else:
            human_possessions = [hex_tile for hex_tile in game_logic.atlas.landscape.values()
                                 if hex_tile.unit and hex_tile.unit.owner == opponent]
            for movable_hex in possessions:
                unit = movable_hex.unit
                if isinstance(unit, City):
                    continue
                target_hex = min(human_possessions,
                                   key=lambda hex_tile: game_logic.atlas.distance(hex_tile, movable_hex))
                if target_hex:
                    self.move_unit_towards_target(unit, movable_hex, target_hex, game_logic)
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
                if neighbor.unit and neighbor.unit.owner == opponent:
                    soldier_targets.add((neighbor, 'soldier', soldier_hex))

        ship_targets = set()
        for ship_hex in ships:
            neighbors = game_logic.atlas.neighbors_within_radius(ship_hex, 2)
            for neighbor in neighbors:
                if neighbor.unit and neighbor.unit.owner == opponent:
                    ship_targets.add((neighbor, 'ship', ship_hex))

        targets = []
        for target_set in [soldier_targets, ship_targets]:
            for target_hex, attack_type, attacker_hex in target_set:
                unit = target_hex.unit
                if isinstance(unit, BattleShip):
                    priority = 1
                elif isinstance(unit, Soldier):
                    priority = 2
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
            if attacker and target_hex.unit:
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
