
import random
from game.atlas import Atlas
from game.player import Human, SimpleAI, ReinforcementAI
from game.entity import City, Soldier, BattleShip, Entity
from game.hex import Hex

class GameLogic:
    def __init__(self, size, mode='human_vs_simpleai'):
        self.atlas = Atlas()
        self.players = []
        
        if mode == 'human_vs_simpleai':
            self.players = [Human("You", (255, 255, 255)), SimpleAI("Opponent", (0, 0, 0))]
        elif mode == 'simpleai_vs_simpleai':
            self.players = [SimpleAI("AI 1", (255, 255, 255)), SimpleAI("AI 2", (0, 0, 0))]
        elif mode == 'reinforcementai_vs_reinforcementai':
            self.players = [ReinforcementAI("Reinforcement AI 1", (255, 255, 255)), ReinforcementAI("Reinforcement AI 2", (0, 0, 0))]
        self.current_player_index = 0
        self.size = size  #Map size
        self.init_map()
        self.place_initial_cities()
        self.game_over = False



    def init_map(self):
        for q in range(-self.size, self.size + 1):
            r1 = max(-self.size, -q - self.size)
            r2 = min(self.size, -q + self.size)
            for r in range(r1, r2 + 1):
                s = -q - r
                terrain = self.random_terrain()
                hex_tile = Hex(q, r, s, terrain)
                self.atlas.add_hex(hex_tile)

    def random_terrain(self):
        p = random.random()
        if p <= 0.36:
            return 'water'
        elif p <= 0.56:
            return 'hill'
        else:
            return 'plain'

    def place_initial_cities(self):
        non_water_tiles = [hex_tile for hex_tile in self.atlas.landscape.values()
                           if not hex_tile.is_water and hex_tile.unit is None]
        player_city_tile = random.choice(non_water_tiles)
        player_city = City(self.players[0])
        player_city_tile.unit = player_city
        self.players[0].units.append(player_city)
        self.players[0].cities.append(player_city)

        #exclude tiles near the first city
        non_water_tiles = [tile for tile in non_water_tiles
                           if self.atlas.distance(tile, player_city_tile) > 0]
        ai_city_tile = random.choice(non_water_tiles)
        ai_city = City(self.players[1])
        ai_city_tile.unit = ai_city
        self.players[1].units.append(ai_city)
        self.players[1].cities.append(ai_city)

    def do_we_have_your_city_near(self, hex_tile, player):
        neighbors = self.atlas.neighbors(hex_tile)
        return any(neighbor.unit and neighbor.unit.is_city and neighbor.unit.owner == player
                   for neighbor in neighbors)

    def got_any_units_near(self, hex_tile, player):
        neighbors = self.atlas.neighbors(hex_tile)
        return any(neighbor.unit and neighbor.unit.owner == player for neighbor in neighbors)

    def build_city(self, player, hex_tile):
        if hex_tile.unit is None and not hex_tile.is_water:
            city = City(player)
            hex_tile.unit = city
            player.units.append(city)
            player.cities.append(city)

    def place_soldier(self, player, hex_tile):
        if hex_tile.unit is None and not hex_tile.is_water:
            soldier = Soldier(player)
            hex_tile.unit = soldier
            player.units.append(soldier)

    def place_battleship(self, player, hex_tile):
        if hex_tile.unit is None and hex_tile.is_water:
            ship = BattleShip(player)
            hex_tile.unit = ship
            player.units.append(ship)

    def move_unit(self, unit, from_hex, to_hex):
        from_hex.unit = None
        to_hex.unit = unit

    def attack_unit(self, attacker, defender):
        if isinstance(attacker, Soldier):
            defender.attacked_by_soldier()
        elif isinstance(attacker, BattleShip):
            defender.attacked_by_ship()
        # Check if defender is defeated
        if defender.hitpoints <= 0:
            defender_hex = self.get_unit_hex(defender)
            if defender_hex:
                defender_hex.unit = None
            defender.owner.units.remove(defender)
            if isinstance(defender, City):
                defender.owner.cities.remove(defender)
                
            if not defender.owner.cities:
                self.game_over = True

    def get_unit_hex(self, unit):
        for hex_tile in self.atlas.landscape.values():
            if hex_tile.unit == unit:
                return hex_tile
        return None

    def next_turn(self):
        if self.game_over:
            return
        current_player = self.players[self.current_player_index]
        #Adjust currency for the current player
        
        current_player.take_turn(self)
        current_player.adjust_currency(3*len(current_player.cities))
        current_player.adjust_currency(-len(current_player.units)) #steady state of 1 city sustaining 2 units.
        self.current_player_index = (self.current_player_index + 1) % 2
        #print(self.current_player_index)


    def distance(self, a, b):
        return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) // 2
    
    def find_path(self, start_hex, goal_hex, unit=None):
        from collections import deque

        frontier = deque()
        frontier.append(start_hex)
        came_from = {}
        came_from[start_hex] = None

        while frontier:
            current = frontier.popleft()
            if current == goal_hex:
                break
            for neighbor in self.atlas.neighbors(current):
                if neighbor not in came_from:

                    if isinstance(unit, Soldier) and neighbor.is_water:
                        continue
                    if isinstance(unit, BattleShip) and not neighbor.is_water:
                        continue
                    if neighbor.unit is None or neighbor == goal_hex:
                        frontier.append(neighbor)
                        came_from[neighbor] = current

        path = []
        current = goal_hex
        while current != start_hex:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []
        path.append(start_hex)
        path.reverse()
        return path