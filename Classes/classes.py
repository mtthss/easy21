
###########
# Imports #
###########
from enum import Enum


###################
# Utility Classes #
###################
class State:

    def __init__(self, dl_curr_sum, pl_curr_sum, is_terminal=False):
        self.pl_sum = pl_curr_sum
        self.dl_sum = dl_curr_sum
        self.term = is_terminal
        self.rew = 0

class Card:

    def __init__(self, color, value):
        self.col = color
        self.val = value


#########
# Enums #
#########
class Actions(Enum):

    # Possible actions
    hit = 0
    stick = 1

    @staticmethod
    def get_action(n):
        return Actions.hit if n==0 else Actions.stick

    @staticmethod
    def get_value(action):
        return 0 if action== Actions.hit else 1

    @staticmethod
    def get_values():
        return [0,1]


class Colors(Enum):

    # Possible card colors
    black = 1
    red = -1
