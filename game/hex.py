from game.entity import BattleShip


class Hex:
    def __init__(self, q, r, s, terrain_type, unit=None):
        assert q + r + s == 0, "Invalid hex coordinates"
        
        assert not (not terrain_type == 'water' and isinstance(unit, BattleShip)), "No battleships on land"
        assert unit is None or not (terrain_type == 'water' and not isinstance(unit, BattleShip)), "No land units on water"
        self.q = q
        self.r = r
        self.s = s
        self.terrain_type = terrain_type  # 'plain', 'hill', 'water'
        self.unit = unit  # Occupying entity

    @property
    def is_water(self):
        return self.terrain_type == 'water'