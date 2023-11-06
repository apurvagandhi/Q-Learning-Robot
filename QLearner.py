""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  

Student Name: Apurva Gandhi
GT User ID: agandhi301
GT ID: 903862828		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import random as rand  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  		 		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False,):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  		 		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		  	   		  		 		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		  		 		  		  		    	 		 		   		 		  
        self.a = 0 
        
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # self.Q = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]  
        # self.R = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]
        # self.Tc = [[[0.00001 for _ in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]	
        # self.T = [[[(1.0 / num_states) for _ in range(num_states)] for _ in range(num_actions)] for _ in range(num_states)]		  		  		    	 		 		   		 		  
        
        # Vectorized for faster performance
        self.Q = np.zeros((num_states, num_actions))
        self.R = np.zeros((num_states, num_actions))
        self.Tc = np.zeros((num_states, num_actions, num_states))
  		  	   		  		 		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		  		 		  		  		    	 		 		   		 		  
        :type s: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        self.s = s  		  	   		  		 		  		  		    	 		 		   		 		  
        action = rand.randint(0, self.num_actions - 1)  		  	   		  		 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		  		 		  		  		    	 		 		   		 		  
            print(f"s = {s}, a = {action}")  		  	   		  		 		  		  		    	 		 		   		 		  
        return action  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  		 		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  		 		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	
        # Update the Q-table for the last state-action pair using Q-learning update rule
	
        self.update_Q_table(self.s, self.a, s_prime, r)
        
        # Hallucination
        if self.dyna != 0:
            self.Tc[self.s, self.a, s_prime] += 1
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + (self.alpha * r)
            for _ in range(self.dyna):
                # Randomly select a state and an action for dyna -- hallucinating
                dyna_s = np.random.randint(0, self.num_states)                
                dyna_a  = np.random.randint(0, self.num_actions)
                # dyna_s = np.random.randint(0, self.num_states, size=self.dyna)
                # dyna_a = np.random.randint(0, self.num_actions, size=self.dyna)
                # Infer s_prime from T by drawing from the one with the highest probability
                dyna_s_prime = np.argmax(self.Tc[dyna_s, dyna_a])
                # Compute dyna r from simulated actions of s and a via R[s, a]
                dyna_r = self.R[dyna_s, dyna_a]
                self.update_Q_table(dyna_s, dyna_a, dyna_s_prime, dyna_r)
           
            # dyna_s_prime = np.argmax(self.Tc[dyna_s, dyna_a], axis = 1)
                
         # Choose the next action 
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # Random action
        else:
            # Pick action with best Q value
            action = np.argmax(self.Q[s_prime])
            
        # Update the exploration rate (rar) with the decay rate (radr)
        self.rar *= self.radr
        # Save the current state and action for the next iteration
        self.s = s_prime
        self.a = action
        
        # Printing for debugging
        if self.verbose:  		  	   		  		 		  		  		    	 		 		   		 		  
            print(f"s = {s_prime}, a = {action}, r={r}") 
             		  	   		  		 		  		  		    	 		 		   		 		  
        return action  	
    
    def update_Q_table (self, s, a, s_prime, r):
        # Old value we used to have that is Q[s,a] + improved estimate that is 
        # Alpha is learning rate - it can take anything between 0 and 1 and usually value is 0.2
        # Gamma is discount rate - it can take anything between 0 and 1 and usually value is 0.9
        #   A low value gamma means that we value later rewards less and high value gamma means we value later reward significantly        
        old_values = (1 - self.alpha) * self.Q[s, a]
        future_discounted_rewards = self.Q[s_prime, np.argmax(self.Q[s_prime])]
        improved_estimates = self.alpha * (r + self.gamma * future_discounted_rewards)
        self.Q[s, a] = old_values + improved_estimates

        return self.Q

  		  	   		  		 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        return "agandhi301"  	
    		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		  		 		  		  		    	 		 		   		 		  
