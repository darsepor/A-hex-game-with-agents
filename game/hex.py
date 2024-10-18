class Hex:
    def __init__(self, q, r, s, terrain_type, unit=None):
        assert q + r + s == 0, "Invalid hex coordinates"
        self.q = q
        self.r = r
        self.s = s
        self.terrain_type = terrain_type  # 'plain', 'hill', 'water'
        self.unit = unit  # Occupying entity

    @property
    def is_water(self):
        return self.terrain_type == 'water'