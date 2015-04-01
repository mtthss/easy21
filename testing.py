
from Environment.environment import Environment
from Agent.agent import Agent
import pickle
import numpy as np
import matplotlib.pyplot as plt


def test_monte_carlo(iterations=1000000, n0=100):
    print "\n-------------------"
    print "Monte Carlo control"
    # learn
    game = Environment()
    agent = Agent(game, n0)
    agent.MC_control(iterations)
    # plot and store
    agent.show_statevalue_function()
    agent.store_Qvalue_function()


def test_sarsa(iterations=1000, mlambda=None, n0=100, avg_it=50):
    print "\n-------------------"
    print "TD control Sarsa"
    monte_carlo_Q = pickle.load(open("Data/Qval_func_1000000_MC_control.pkl", "rb"))
    n_elements = monte_carlo_Q.shape[0]*monte_carlo_Q.shape[1]*2
    mse = []
    sse = []

    if not isinstance(mlambda,list):
        # if no value is passed for lambda, default 0.5
        l = 0.5 if mlambda==None else mlambda
        # learn
        game = Environment()
        agent = Agent(game, n0)
        agent.TD_control(iterations, l, avg_it)
        # plot results
        agent.show_statevalue_function()
    else:
        # test each value of lambda
        for l in mlambda:
            game = Environment()
            agent = Agent(game, n0)
            l_mse = agent.TD_control(iterations, l, avg_it)
            mse.append(l_mse)

        plt.plot(mlambda,mse)
        plt.ylabel('mse')
        plt.show()


def test_linear_sarsa(iterations=1000, mlambda=None, n0=100, avg_it=100):
    print "\n-------------------"
    print "TD control Sarsa, Linear function approximation"
    monte_carlo_Q = pickle.load(open("Data/Qval_func_1000000_MC_control.pkl", "rb"))
    n_elements = monte_carlo_Q.shape[0]*monte_carlo_Q.shape[1]*2
    mse = []
    sse = []
    if not isinstance(mlambda,list):
        # if no value is passed for lambda, default 0.5
        l = 0.5 if mlambda==None else mlambda
        # learn
        game = Environment()
        agent = Agent(game, n0)
        agent.TD_control_linear(iterations,l,avg_it)
        agent.show_statevalue_function()
    else:
        # test each value of lambda
        for l in mlambda:
            game = Environment()
            agent = Agent(game, n0)
            l_mse = agent.TD_control_linear(iterations,l,avg_it)
            mse.append(l_mse)

        plt.plot(mlambda,mse)
        plt.ylabel('mse')
        plt.show()

if __name__ == '__main__':

    # parameters
    lambdas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    iterationsMC = 1000000
    iterationsSRS = 1000
    n0 = 100

    # testing
    #test_monte_carlo(iterationsMC,n0)
    test_sarsa(iterationsSRS,lambdas,n0, avg_it=10)
    #test_linear_sarsa(iterationsSRS,lambdas,n0, avg_it=5)