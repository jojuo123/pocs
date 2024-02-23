import torch
import numpy as np
import pdb
# from memory_profiler import profile
import time
import os
import psutil
import tracemalloc
from bisect import bisect_right

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
# decorator function
# def profile(func):
#     def wrapper(*args, **kwargs):
 
#         mem_before = process_memory()
#         result = func(*args, **kwargs)
#         mem_after = process_memory()
#         print("{}:consumed memory: {:,}".format(
#             func.__name__,
#             mem_before, mem_after, mem_after - mem_before))
 
#         return result
#     return wrapper

def timing(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        # mem_before = process_memory()
        # tracemalloc.start()
        result = func(*args, **kwargs)
        # mem_after = tracemalloc.get_traced_memory()
        # mem_after = process_memory()
        # tracemalloc.stop()
        end = time.perf_counter_ns()
        # print('time consume: ', end-start)
        return result, end-start, 0
    return wrapper

class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    Parameters:
    -----------
    
    - S: Number of states.
    - T: numpy.array Transition matrix of size S by S
         stores probability from state i to state j.
    - E: numpy.array Emission matrix of size S by N (number of observations)
         stores the probability of observing  O_j  from state  S_i. 
    - T0: numpy.array Initial state probabilities of size S.
    """

    def __init__(self, T, E, T0, epsilon = 0.001, step = 10):
        # Max number of iteration
        self.step = step
        # convergence criteria
        self.epsilon = epsilon 
        # Number of possible states
        self.S = T.shape[0]
        # Number of possible observations
        self.O = E.shape[0]
        self.prob_state_1 = []
        # Emission probability
        self.E = torch.tensor(E)
        # Transition matrix
        self.T = torch.tensor(T)
        # Initial state vector
        self.T0 = torch.tensor(T0)
    
    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)
        self.posterior = torch.zeros_like(self.forward)
        
    def _forward(model, obs_prob_seq):
        model.scale = torch.zeros([model.N], dtype=torch.float64) #scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()
        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob
         # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0)
            # transition prior
            prior_prob = torch.matmul(prev_prob, model.T)
            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)
            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1] * forward_prob
    
    def _backward(self, obs_prob_seq_rev):
        # initialize with state ending priors
        self.backward[0] = self.scale[self.N - 1] * torch.ones([self.S], dtype=torch.float64)
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[self.N - 2 - step] * backward_prob
        self.backward = torch.flip(self.backward, [0, 1])
    @timing
    def forward_backward(self, obs_prob_seq):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """        
        self.initialize_forw_back_variables([obs_prob_seq.shape[0], self.S])
        self._forward(obs_prob_seq)
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        # self._backward(obs_prob_seq_rev)

    r'''
        checkpoint implementation starts here
    '''
    def initialize_forw_back_variables_checkpoint(self, shape, step):
        # self.forward = torch.zeros(shape, dtype=torch.float64)
        # self.backward = torch.zeros_like(self.forward)
        # self.posterior = torch.zeros_like(self.forward)
        self.checkpoints = []
        for i in range(0, shape[0], step):
            self.checkpoints.append(i)
        if self.checkpoints[-1] != shape[0] - 1:
            self.checkpoints.append(shape[0] - 1)
        
        self.checkpoints2index = {ckpt : i for i, ckpt in enumerate(self.checkpoints)}

        self.n_checkpoints = len(self.checkpoints)
        # print(self.n_checkpoints)
        self.forward = torch.zeros((self.n_checkpoints, shape[1]), dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)
        self.posterior = torch.zeros_like(self.forward)

    def prev_checkpoint_index_search(self, checkpoints, index):
        return bisect_right(self.checkpoints, index)

    def _forward_from_checkpoint(self, init_prob, obs_prob_seq):
        prev_prob = init_prob
        scale = 1.0
        for step, obs_prob in enumerate(obs_prob_seq):
            prior_prob = torch.matmul(prev_prob, self.T)
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)
            scale = 1 / forward_prob.sum()
            prev_prob = scale * forward_prob
        return prev_prob, scale
        
    def _forward_checkpoint(model, obs_prob_seq):
        model.scale = torch.zeros([model.n_checkpoints], dtype=torch.float64) #scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()
        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob
         # propagate belief
        # for step, obs_prob in enumerate(obs_prob_seq[1:]):
        #     prev_checkpoint_index = model.prev_checkpoint_index_search(model.checkpoints, step+1)
        #     # previous state probability
        #     # prev_prob = model.forward[step].unsqueeze(0)
        #     prev_prob = model.forward[prev_checkpoint_index]
        #     start_t = model.checkpoints[prev_checkpoint_index]
        #     prev_prob = model._forward_from_checkpoint(prev_prob, obs_prob_seq[start_t+1:step+1])
        #     # transition prior
        #     prior_prob = torch.matmul(prev_prob, model.T)
        #     # forward belief propagation
        #     forward_score = prior_prob * obs_prob
        #     forward_prob = torch.squeeze(forward_score)
        #     # scaling factor
        #     model.scale[step + 1] = 1 / forward_prob.sum()
        #     # Update forward matrix
        #     model.forward[step + 1] = model.scale[step + 1] * forward_prob

        for i, ckpt in enumerate(model.checkpoints[1:], 1):
            prev_checkpoint_start = model.checkpoints[i-1]
            prev_checkpoint_prob = model.forward[i-1]
            fwd_prob, scale = model._forward_from_checkpoint(prev_checkpoint_prob, obs_prob_seq[prev_checkpoint_start+1:ckpt])
            model.forward[i] = fwd_prob 
            model.scale[i] = scale 

    def _backward_from_checkpoint(self, init_prob, obs_prob_seq):
        next_prob = init_prob
        for step, obs_prob in reversed(list(enumerate(obs_prob_seq))):
            obs_prob_d = torch.diag(obs_prob)
            prior_prob = torch.matmul(self.T, obs_prob_d)
            next_prob = torch.matmul(prior_prob, next_prob).squeeze()
            next_prob = self.scale
    
    def _backward_checkpoint(self, obs_prob_seq_rev):
        # initialize with state ending priors
        self.backward[self.n_checkpoints - 1] = self.scale[self.n_checkpoints - 1] * torch.ones([self.S], dtype=torch.float64)
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[self.N - 2 - step] * backward_prob
        self.backward = torch.flip(self.backward, [0, 1])

        for i, ckpt in reversed(list(enumerate(self.checkpoints[:-1]))):
            next_checkpoint_start = self.checkpoints[i+1]
            next_checkpoint_prob = self.backward[i+1]



    @timing
    def forward_backward_checkpoint(self, obs_prob_seq):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """        
        self.initialize_forw_back_variables_checkpoint([obs_prob_seq.shape[0], self.S], self.step)
        self._forward_checkpoint(obs_prob_seq)
        # obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        # self._backward_checkpoint(obs_prob_seq_rev)
