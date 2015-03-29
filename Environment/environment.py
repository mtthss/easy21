
###########
# Imports #
###########
from Classes.classes import State, Actions, Colors, Card
from Agent.agent import Agent
import random


###############
# Environment #
###############
class Environment:

    def __init__(self):
        self.card_min = 1    # min absolute val of card
        self.card_max = 10   # max absolute val of card
        self.dl_values = 10  # possible values for dl in state
        self.pl_values = 21  # possible values for pl in state
        self.act_values = len(Actions.get_values())  # number of possible actions

    def get_initial_state(self):

        # First round: one black card each
        dl_first_card = self.draw_black_card()
        pl_first_card = self.draw_black_card()
        s0 = State(dl_first_card.val, pl_first_card.val)
        return s0

    def step(self, s, a):

        # Execute Action
        if a==Actions.hit:
            # Player draws a card
            c = self.draw_card()
            v = c.val if c.col==Colors.black else -c.val
            s2 = State(s.dl_sum,s.pl_sum+v)
            # Evaluate if the player "goes bust"
            if s2.pl_sum<1 or s2.pl_sum>21:
                s2.term = True
                s2.rew = -1
        # Dealer makes his moves
        else:
            # Initialize
            s2 = State(s.dl_sum,s.pl_sum)
            draw_again = True
            # Iterate until Stick or Bust
            while draw_again:
                c = self.draw_card()
                v = c.val if c.col==Colors.black else -c.val
                s2.dl_sum = s2.dl_sum+v
                # Dl busts
                if s2.dl_sum<1 or s2.dl_sum>21:
                    s2.rew = 1
                    draw_again = False
                # Dl sticks
                elif s2.dl_sum > 16:
                    # Pl wins
                    if s2.dl_sum < s2.pl_sum:
                        s2.rew = 1
                    # Draw
                    elif s2.dl_sum == s2.pl_sum:
                        s2.rew = 0
                    # Dl wins
                    else:
                        s2.rew = -1
                    draw_again = False
            s2.term = True

        # Return state (including current reward)
        return s2

    def draw_card(self):

        # draw a card, black with probability 2/3
        if random.random() < 2.0/3:
            return self.draw_black_card()
        else:
            return Card(Colors.red,random.randint(self.card_min,self.card_max))

    def draw_black_card(self):

        # draw a black card
        return Card(Colors.black,random.randint(self.card_min,self.card_max))

########
# Test #
########

game = Environment()
agent = Agent(game, 100)
agent.TD_control(1000000,0.5)

agent.show_statevalue_function()

game = Environment()
agent = Agent(game, 100)
agent.MC_control(1000000)

agent.show_statevalue_function()
agent.show_previous_statevalue_function("../Data/val_func_1000000_MC_control.pkl")