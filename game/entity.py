
class Entity:
    city_cost = 10
    ship_cost = 10
    soldier_cost = 1

    def __init__(self, owner):
        self.owner = owner
        self.hitpoints = 1

    def attacked_by_soldier(self): #Units significantly more vulnerable if in debt. Adds strategic depth.
        self.hitpoints -= 1
        if self.owner.currency < 0:
            self.hitpoints -=2

    def attacked_by_ship(self):
        self.hitpoints -= 1.5
        if self.owner.currency < 0:
            self.hitpoints -=1.5

    @property
    def is_city(self):
        return False


class City(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 3

    @property
    def is_city(self):
        return True


class Soldier(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 3


class BattleShip(Entity):
    def __init__(self, owner):
        super().__init__(owner)
        self.hitpoints = 1
