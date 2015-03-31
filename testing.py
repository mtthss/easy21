
from Environment.environment import Environment
from Agent.agent import Agent


def test_monte_carlo(iterations=1000000, n0=100):
    print "\nMonte Carlo control"
    game = Environment()
    agent = Agent(game, n0)
    agent.MC_control(iterations)
    agent.show_statevalue_function()


def test_sarsa(iterations=1000, mlambda=None, n0=100):
    print "\nTD control Sarsa"
    game = Environment()
    agent = Agent(game, n0)
    agent.TD_control(iterations, mlambda)
    agent.show_statevalue_function()


def test_linear_sarsa(iterations=1000, mlambda=None, n0=100):
    print "\nTD control Sarsa, Linear function approximation"
    game = Environment()
    agent = Agent(game, n0)
    agent.TD_control_linear(iterations,mlambda)
    agent.show_statevalue_function()


if __name__ == '__main__':

    #test_monte_carlo(1000000,100)
    #test_sarsa(1000,0.5,100)
    test_linear_sarsa(1000,0.5,100)
