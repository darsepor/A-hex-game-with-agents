
import pygame
import math
from game.game_logic import GameLogic
from game.entity import Entity, City, Soldier, BattleShip
from game.hex import Hex

class GameGUI:
    def __init__(self, mode='human_vs_simpleai'):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("Hex Game")
        self.clock = pygame.time.Clock()
        self.game_logic = GameLogic(mode=mode)
        self.size = 30  #Hexagon radius
        self.center = (500, 500)
        self.running = True
        self.selected_unit = None
        self.selected_hex = None
        self.highlighted_tiles = []
        self.action_mode = None  #move or attack
        self.context_menu = None
        self.font = pygame.font.SysFont('Arial', 18, bold=False)
        self.mode = mode

    def run(self):
        while self.running:
            if self.game_logic.game_over:
                self.running = False
                break

            current_player = self.game_logic.players[self.game_logic.current_player_index]
            self.handle_events()
            if current_player.is_ai:
                #AI's turn
                self.game_logic.next_turn()
                pygame.time.delay(400)
            else:
                pass #we're already handling events above, if it's two AI's input does nothing (but there has
                     #to be input such that the program is not unresponsive and does not crash etc.)
                

            self.draw()
            pygame.display.flip()
            self.clock.tick(10)

    def hex_to_pixel(self, q, r):
        #Pointy-top hexagon orientation
        x = self.size * (math.sqrt(3) * q + (math.sqrt(3) / 2) * r) + self.center[0]
        y = self.size * ((3 / 2) * r) + self.center[1]
        return (x, y)

    def pixel_to_hex(self, x, y):
        x -= self.center[0]
        y -= self.center[1]
        q = ((math.sqrt(3) / 3) * x - (1 / 3) * y) / self.size
        r = ((2 / 3) * y) / self.size
        return self.cube_round(q, r, -q - r)

    def cube_round(self, q, r, s):
        rq = round(q)
        rr = round(r)
        rs = round(s)

        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)

        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        else:
            rs = -rq - rr
        return (rq, rr, rs)

    def get_hex_at_pixel(self, x, y):
        q, r, s = self.pixel_to_hex(x, y)
        return self.game_logic.atlas.get_hex(q, r, s)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.mode=='human_vs_simpleai':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if self.context_menu:
                            self.handle_context_menu_click(event.pos)
                        else:
                            hex_tile = self.get_hex_at_pixel(*event.pos)
                            if hex_tile:
                                self.handle_left_click(hex_tile)
                    elif event.button == 3:  # Right click
                        hex_tile = self.get_hex_at_pixel(*event.pos)
                        if hex_tile:
                            self.handle_right_click(hex_tile, event.pos)


    def handle_left_click(self, hex_tile):
        player = self.game_logic.players[self.game_logic.current_player_index]
        if self.action_mode == 'move':
            if hex_tile in self.highlighted_tiles and hex_tile.unit is None:
                self.game_logic.move_unit(self.selected_unit, self.selected_hex, hex_tile)
                self.end_action()
                self.game_logic.next_turn()
            else:
                self.end_action()
        elif self.action_mode == 'attack':
            if hex_tile in self.highlighted_tiles and hex_tile.unit and hex_tile.unit.owner != player:
                self.game_logic.attack_unit(self.selected_unit, hex_tile.unit)
                self.end_action()
                self.game_logic.next_turn()
            else:
                self.end_action()
        else:
            if hex_tile.unit and hex_tile.unit.owner == player:

                self.selected_unit = hex_tile.unit
                self.selected_hex = hex_tile
                self.action_mode = 'move'
                self.highlight_movement_range()
            else:
                self.end_action()

    def handle_right_click(self, hex_tile, mouse_pos):
        player = self.game_logic.players[self.game_logic.current_player_index]
        if hex_tile.unit and hex_tile.unit.owner == player:

            self.selected_unit = hex_tile.unit
            self.selected_hex = hex_tile
            self.action_mode = 'attack'
            self.highlight_attack_range()
        else:
            #Show context menu
            self.handle_right_click_menu(hex_tile, mouse_pos)

    def handle_right_click_menu(self, hex_tile, mouse_pos):
        player = self.game_logic.players[self.game_logic.current_player_index]
        available_actions = []

        if hex_tile.unit is None:
            if hex_tile.is_water:
                if self.game_logic.do_we_have_your_city_near(hex_tile, player):
                    available_actions.append('Place Battleship')
            else:
                if self.game_logic.do_we_have_your_city_near(hex_tile, player):
                    available_actions.append('Recruit Soldier')
                if self.game_logic.got_any_units_near(hex_tile, player):
                    available_actions.append('Build City')

        if available_actions:
            self.show_context_menu(mouse_pos, available_actions, hex_tile)

    def highlight_movement_range(self):
        self.highlighted_tiles = []
        if self.selected_unit:
            current_hex = self.selected_hex
            if isinstance(self.selected_unit, Soldier):
                radius = 1
                tiles = self.game_logic.atlas.neighbors_within_radius(current_hex, radius)
                self.highlighted_tiles = [tile for tile in tiles if not tile.is_water and tile.unit is None]
            elif isinstance(self.selected_unit, BattleShip):
                radius_one = self.game_logic.atlas.neighbors_within_radius(current_hex, 1)
                radius_two = []
                for tile in radius_one:
                    if tile.is_water and tile.unit is None:
                        neighbors = self.game_logic.atlas.neighbors_within_radius(tile, 1)
                        radius_two.extend([t for t in neighbors if t.is_water and t.unit is None])
                # Ensure tiles are water and unoccupied
                self.highlighted_tiles = [tile for tile in set(radius_one + radius_two) if tile.is_water and tile.unit is None]

    def highlight_attack_range(self):
        self.highlighted_tiles = []
        if self.selected_unit:
            current_hex = self.selected_hex
            if isinstance(self.selected_unit, Soldier):
                radius = 1
            elif isinstance(self.selected_unit, BattleShip):
                radius = 2
            else:
                radius = 0
            tiles = self.game_logic.atlas.neighbors_within_radius(current_hex, radius)
            self.highlighted_tiles = [tile for tile in tiles if tile.unit and tile.unit.owner != self.selected_unit.owner]

    def end_action(self):
        self.selected_unit = None
        self.selected_hex = None
        self.highlighted_tiles = []
        self.action_mode = None

    def show_context_menu(self, mouse_pos, actions, hex_tile):
        self.context_menu = {
            'position': mouse_pos,
            'actions': actions,
            'hex_tile': hex_tile
        }

    def handle_context_menu_click(self, pos):
        x, y = pos
        menu_x, menu_y = self.context_menu['position']
        actions = self.context_menu['actions']
        menu_width = 150
        menu_height = 20 * len(actions)
        if menu_x <= x <= menu_x + menu_width and menu_y <= y <= menu_y + menu_height:
            idx = (y - menu_y) // 20
            if 0 <= idx < len(actions):
                action = actions[int(idx)]
                self.execute_action(action, self.context_menu['hex_tile'])
        self.context_menu = None  # Close the menu

    def execute_action(self, action, hex_tile):
        player = self.game_logic.players[self.game_logic.current_player_index]
        if action == 'Build City':
            cost = Entity.city_cost
            if player.currency >= cost:
                self.game_logic.build_city(player, hex_tile)
                player.adjust_currency(-cost)
                self.game_logic.next_turn()
        elif action == 'Recruit Soldier':
            cost = Entity.soldier_cost
            if player.currency >= cost:
                self.game_logic.place_soldier(player, hex_tile)
                player.adjust_currency(-cost)
                self.game_logic.next_turn()
        elif action == 'Place Battleship':
            cost = Entity.ship_cost
            if player.currency >= cost:
                self.game_logic.place_battleship(player, hex_tile)
                player.adjust_currency(-cost)
                self.game_logic.next_turn()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_hexes()
        self.draw_units()
        self.draw_ui()
        if self.context_menu:
            self.draw_context_menu()

    def draw_hexes(self):
        for hex_tile in self.game_logic.atlas.landscape.values():
            x, y = self.hex_to_pixel(hex_tile.q, hex_tile.r)
            corners = self.get_hex_corners(x, y)
            if hex_tile in self.highlighted_tiles:
                if self.action_mode == 'move':
                    color = (173, 216, 230)  #Light Blue for movement
                elif self.action_mode == 'attack':
                    color = (255, 182, 193)  #Pink for attack
            else:
                if hex_tile.terrain_type == 'plain':
                    color = (144, 238, 144)  #Green
                elif hex_tile.terrain_type == 'hill':
                    color = (169, 169, 169)  #Gray
                else:
                    color = (224, 255, 255)  #Cyan
            pygame.draw.polygon(self.screen, color, corners)
            pygame.draw.polygon(self.screen, (0, 0, 0), corners, 1)

    def get_hex_corners(self, x, y):
        #Pointy-top hexagon
        corners = []
        for i in range(6):
            angle = 2 * math.pi * (i + 0.5) / 6
            dx = self.size * math.cos(angle)
            dy = self.size * math.sin(angle)
            corners.append((x + dx, y + dy))
        return corners

    def draw_units(self):
        for hex_tile in self.game_logic.atlas.landscape.values():
            if hex_tile.unit:
                x, y = self.hex_to_pixel(hex_tile.q, hex_tile.r)
                if isinstance(hex_tile.unit, City):
                    self.draw_city(hex_tile.unit, x, y)
                elif isinstance(hex_tile.unit, Soldier):
                    self.draw_soldier(hex_tile.unit, x, y)
                elif isinstance(hex_tile.unit, BattleShip):
                    self.draw_battleship(hex_tile.unit, x, y)

    def draw_city(self, city, x, y):
        color = city.owner.color
        pygame.draw.circle(self.screen, color, (int(x), int(y)), 15)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 15, 1)
        self.draw_unit_health(city, x, y, color)

    def draw_soldier(self, soldier, x, y):
        color = soldier.owner.color
        rect = pygame.Rect(x - 10, y - 10, 20, 20)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        self.draw_unit_health(soldier, x, y, color)

    def draw_battleship(self, ship, x, y):
        color = ship.owner.color
        width_top = 20
        width_bottom = 45
        height = 20

        points = [
            (x - width_top / 2, y - height / 2),
            (x + width_top / 2, y - height / 2),
            (x + width_bottom / 2, y + height / 2),
            (x - width_bottom / 2, y + height / 2)
        ]

        pygame.draw.polygon(self.screen, color, points)

        pygame.draw.polygon(self.screen, (0, 0, 0), points, 1)
        self.draw_unit_health(ship, x, y, color)

        
    def draw_unit_health(self, unit, x, y, unit_color):
        text_color = (255, 255, 255) if unit_color[0]==0 else (0, 0, 0)

        health_text = self.font.render(str(unit.hitpoints), True, text_color)
        text_rect = health_text.get_rect(center=(x, y))
        self.screen.blit(health_text, text_rect)

    def draw_ui(self):

        y_offset = 10
        title_text = self.font.render("Gold Balances:", True, (0, 0, 0))
        self.screen.blit(title_text, (10, y_offset))
        y_offset += 25

        for player in self.game_logic.players:
            color_rect = pygame.Surface((20, 20))
            color_rect.fill(player.color)
            self.screen.blit(color_rect, (10, y_offset))

            player_text = self.font.render(f"{player.name}: {player.currency}", True, (0, 0, 0))
            self.screen.blit(player_text, (35, y_offset + 2))
            y_offset += 25
        
        
        
    def draw_context_menu(self):
        x, y = self.context_menu['position']
        actions = self.context_menu['actions']
        menu_width = 150
        menu_height = 20 * len(actions)
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, menu_width, menu_height))
        for idx, action in enumerate(actions):
            action_text = self.font.render(action, True, (0, 0, 0))
            self.screen.blit(action_text, (x + 5, y + idx * 20))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, menu_width, menu_height), 1)


if __name__ == "__main__":
    gui = GameGUI(mode='human_vs_simpleai')
    gui.run()
