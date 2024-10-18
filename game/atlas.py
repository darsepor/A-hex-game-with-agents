
import math

class Atlas:
    def __init__(self):
        self.landscape = {}  #Key: (q, r, s), Value: Hex

    def add_hex(self, hex_tile):
        self.landscape[(hex_tile.q, hex_tile.r, hex_tile.s)] = hex_tile

    def get_hex(self, q, r, s):
        return self.landscape.get((q, r, s))

    def neighbors(self, hex_tile):
        directions = [
            (+1, -1, 0), (+1, 0, -1), (0, +1, -1),
            (-1, +1, 0), (-1, 0, +1), (0, -1, +1)
        ]
        result = []
        for dq, dr, ds in directions:
            neighbor = self.get_hex(hex_tile.q + dq, hex_tile.r + dr, hex_tile.s + ds)
            if neighbor:
                result.append(neighbor)
        return result

    def distance(self, a, b):
        return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) // 2

    def neighbors_within_radius(self, tile, radius):
        qt, rt, st = tile.q, tile.r, tile.s
        result = []
        for (q, r, s), hex_tile in self.landscape.items():
            dq = abs(q - qt)
            dr = abs(r - rt)
            ds = abs(s - st)
            max_diff = max(dq, dr, ds)
            if max_diff <= radius:
                result.append(hex_tile)
        return result