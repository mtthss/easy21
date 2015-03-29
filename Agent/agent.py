
###########
# Imports #
###########
from Classes.classes import Actions
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from mpl_toolkits.mplot3d import Axes3D


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
        self.method = ""

        # initialize tables for (state, action) pairs occurrences, values, eligibility
        self.N = np.zeros((self.env.dl_values, self.env.pl_values, self.env.act_values))
        self.Q = np.zeros((self.env.dl_values, self.env.pl_values, self.env.act_values))
        self.E = np.zeros((self.env.dl_values, self.env.pl_values, self.env.act_values))
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
            return Actions.get_action(np.argmax(self.Q[state.dl_sum-1, state.pl_sum-1, :]))


    # play specified number of games, learning from experience using Monte-Carlo Control
    def MC_control(self, iterations):

        # Initialise
        self.iter = iterations
        self.method = "MC_control"
        count_wins = 0
        episode_pairs = []

        # Loop over episodes (complete game runs)
        for episode in xrange(self.iter):

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
                error = my_state.rew - self.Q[curr_s.dl_sum-1, curr_s.pl_sum-1, Actions.get_value(curr_a)]
                self.Q[curr_s.dl_sum-1, curr_s.pl_sum-1, Actions.get_value(curr_a)] += step * error

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
                self.V[d,p] = max(self.Q[d, p, :])


    # play specified number of games, learning from experience using TD Control (Sarsa)
    def TD_control(self, iterations, mlambda):

        self.mlambda = float(mlambda)
        self.iter = iterations
        self.method = "Sarsa_control"

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
                alpha = 1.0  / (self.N[s.dl_sum-1, s.pl_sum-1,  Actions.get_value(a)])
                try:
                    delta = s_next.rew + self.Q[s_next.dl_sum-1, s_next.pl_sum-1, Actions.get_value(a_next)] \
                        - self.Q[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)]
                except:
                    delta = s_next.rew - self.Q[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)]
                self.E[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)] += 1
                update = alpha*delta*self.E
                self.Q = self.Q[s.dl_sum-1, s.pl_sum-1, Actions.get_value(a)]+update
                self.E = self.mlambda*self.E

                # reassign s and a
                s = s_next
                a = a_next

            if episode%10000==0: print "Episode: %d, Reward: %d" %(episode, s_next.rew)
            count_wins = count_wins+1 if s_next.rew==1 else count_wins

        print float(count_wins)/self.iter*100

        # Derive value function
        for d in xrange(self.env.dl_values):
            for p in xrange(self.env.pl_values):
                self.V[d,p] = max(self.Q[d, p, :])


    # compute feature
    def feature_computation(self, state, action):

        d_edges = [[1,4],[4,7],[7,10]]
        p_edges = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]]
        actions = [0,1]

        feature_vect = np.zeros(len(d_edges)*len(p_edges)*2)


    # store in a txt file
    def store_statevalue_function(self):

        with open('../Data/results.csv', 'wb') as csvout:

            write_out = csv.writer(csvout, delimiter = ',')
            for row in self.V:
                write_out.writerow(row)


    # plot value function learnt
    def show_statevalue_function(self):

        def get_stat_val(x,y):
            return self.V[x,y]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = np.arange(0, self.env.dl_values-1, 1)
        Y = np.arange(0, self.env.pl_values-1, 1)
        X,Y = np.meshgrid(X,Y)
        Z = get_stat_val(X,Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()

        my_method = self.method
        my_iterations = str(self.iter)
        pickle.dump(self.V, open("../Data/val_func_%s_%s.pkl" %(my_iterations, my_method), "wb"))


    def show_previous_statevalue_function(self, path):
        V = pickle.load(open(path,"rb"))

        def get_stat_val(x,y):
            return V[x,y]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = np.arange(0, self.env.dl_values-1, 1)
        Y = np.arange(0, self.env.pl_values-1, 1)
        X,Y = np.meshgrid(X,Y)

        Z = get_stat_val(X,Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()


