
###########
# Imports #
###########
from Classes.classes import Actions
import random
import numpy as np
import csv
import matplotlib.pyplot as plt

#########
# Agent #
#########
class Agent:

    # Constructor, initialize attributes
    def __init__(self, environment, n0):

        # initialize variables
        self.iter = 0
        self.n0 = float(n0)
        self.env = environment

        # initialize table for counting action-state pairs occurrences
        self.N = np.zeros((self.env.dl_values, self.env.pl_values, self.env.act_values))

        # initialize action value function lookup table
        self.AV = np.zeros((self.env.dl_values, self.env.pl_values, self.env.act_values))

        # initialize value function
        self.V = np.zeros((self.env.dl_values, self.env.pl_values))


    # get optimal action, with epsilon exploration (epsilon dependent on number of visits to the state)
    def eps_greedy_choice(self, state):

        # collect visits
        try:
            visits_to_state = sum(self.N[state.dl_sum-1, state.pl_sum-1, :])
        except:
            visits_to_state = 0

        # compute epsilon
        curr_epsilon = self.n0 / (self.n0 + visits_to_state)

        # epsilon greedy policy
        if random.random() < curr_epsilon:
            return Actions.hit if random.random()<0.5 else Actions.stick
        else:
            return Actions.get_action(np.argmax(self.AV[state.dl_sum-1, state.pl_sum-1, :]))


    # play specified number of games, learning from experience using Monte-Carlo Control
    def MC_control(self, iterations):

        # Initialise
        self.iter = iterations
        count_wins = 0
        episode_pairs = []

        # Loop over episodes (complete game runs)
        for episode in xrange(self.iter):

            ###################################################
            # fai lista triplette visitate in ciascun episode #
            # cosi eviti di attraversare tutta matrice        #
            ###################################################

            # reset state action pair list
            episode_pairs = []

            # get initial state for current episode
            my_state = self.env.get_initial_state()

            # Execute until game ends
            while not my_state.term:

                # choose action with epsilon greedy policy
                my_action = self.eps_greedy_choice(my_state)

                # store action state pairs
                episode_pairs.append((my_state, my_action))

                # update visits
                self.N[my_state.dl_sum-1, my_state.pl_sum-1, Actions.get_value(my_action)] += 1

                # execute action
                my_state = self.env.step(my_state, my_action)

            # Update Action value function accordingly
            for curr_s, curr_a in episode_pairs:
                step = 1.0  / (self.N[curr_s.dl_sum-1, curr_s.pl_sum-1, Actions.get_value(curr_a)])
                error = my_state.rew - self.AV[curr_s.dl_sum-1, curr_s.pl_sum-1, Actions.get_value(curr_a)]
                self.AV[curr_s.dl_sum-1, curr_s.pl_sum-1, Actions.get_value(curr_a)] += step * error

            """
            for d in xrange(self.env.dl_values):
                for p in xrange(self.env.pl_values):
                    for a in xrange(self.env.act_values):
                        if not self.N[d, p, a]==0:
                            step = 1.0  / (self.N[d, p, a])
                            self.AV[d, p, a] += step * (my_state.rew - self.AV[d, p, a])
            """
            if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, my_state.rew)
            count_wins = count_wins+1 if my_state.rew==1 else count_wins

        print count_wins

        # Derive value function
        for d in xrange(self.env.dl_values):
            for p in xrange(self.env.pl_values):
                self.V[d,p] = max(self.AV[d, p, :])


    # play specified number of games, learning from experience using TD Control (Sarsa)
    def TD_control(self, iterations, mlambda):

        self.mlambda = float(mlambda)
        self.iter = iterations

        count_wins = 0

        # Loop over episodes (complete game runs)
        for episode in xrange(self.iter):

            s = self.env.get_initial_state()
            a = self.eps_greedy_choice(s)

            # Execute until game ends
            while not s.term:

                # update visit count
                self.N[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)] += 1

                # execute action
                s_next = self.env.step(s, a)

                # choose next action with epsilon greedy policy
                a_next = self.eps_greedy_choice(s_next)

                # update action value function
                step = 1.0  / (self.N[s.dl_sum-1, s.pl_sum-1,  Actions.get_value(a)])
                try:
                    new = self.AV[s_next.dl_sum-1, s_next.pl_sum-1, Actions.get_value(a_next)]
                except:
                    new = 0
                old = self.AV[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)]
                self.AV[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)] += step * (s_next.rew + new - old)

                # reassign s and a
                s = s_next
                a= a_next

            if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, s_next.rew)
            count_wins = count_wins+1 if s_next.rew==1 else count_wins

        print count_wins

        # Derive value function
        for d in xrange(self.env.dl_values):
            for p in xrange(self.env.pl_values):
                self.V[d,p] = max(self.AV[d, p, :])


    # store in a txt file
    def store_statevalue_function(self):

        with open('../Data/results.csv', 'wb') as csvout:

            write_out = csv.writer(csvout, delimiter = ',')
            for row in self.V:
                write_out.writerow(row)



    # plot value function learnt
    def show_statevalue_function(self):

        x = np.linspace(0, 1, self.env.pl_values)
        y = self.V[0,:]
        plt.figure()
        plt.plot(x, y, 'r')

        x = np.linspace(0, 1, self.env.pl_values)
        y = self.V[1,:]
        plt.plot(x, y, 'g')

        x = np.linspace(0, 1, self.env.pl_values)
        y = self.V[2,:]
        plt.plot(x, y, 'b')

        x = np.linspace(0, 1, self.env.pl_values)
        y = self.V[2,:]
        plt.plot(x, y, 'k')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('title')
        plt.show()
