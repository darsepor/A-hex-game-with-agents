
class Entity:
    city_cost = 100
    ship_cost = 30
    soldier_cost = 30

    def __init__(self, owner):
        self.owner = owner
        self.hitpoints = 1

    def attacked_by_soldier(self):
        self.hitpoints -= 10

    def attacked_by_ship(self):
        self.hitpoints -= 20

    @property
    def is_city(self):
        return False


class City(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 40

    @property
    def is_city(self):
        return True


class Soldier(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 30


class BattleShip(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 10
