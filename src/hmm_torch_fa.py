import torch
import numpy as np
import pdb
# from memory_profiler import profile
import time
import os
import psutil
import tracemalloc
from tqdm import tqdm

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

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

def evaluate_GMM(states, id2int, obs, allmeans, allvar, allprob, mask, id2state): 
        
    ''' evaluate emission probabilities with multivariate GMM 
    params: 
        state: hidden state of interest 
        obs: observation of interest 
        
    return 
        density
    
    ''' 
    densities = torch.zeros((len(states)), dtype=torch.float64)
    for st in states:
        int_st = id2int[st]
        if not mask[int_st]:
            continue
        st = id2state[st]
        means = torch.tensor(allmeans[st]) 
        variances = torch.tensor(allvar[st]) 
        weights = [float(x.strip()) for x in allprob[st]]
        density = 0 
        # print(obs)
        # print(means.shape, variances.shape, len(weights))
        for i in range(len(weights)): 
            # var = multivariate_normal(means[i], cov=np.diag(variances[i]))
            # print(means[i].shape, variances[i].shape)
            var = torch.distributions.multivariate_normal.MultivariateNormal(loc=means[i], covariance_matrix=torch.diag(variances[i]))
            # density += (np.log(weights[i]) + np.log(var.pdf(obs)))
            # print(var.log_prob(obs))
            density += weights[i] * torch.exp(var.log_prob(obs))
            # print(torch.exp(var.log_prob(obs)))
        densities[int_st] = density.item()
    return densities


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

    # def __init__(self, T, E, T0, epsilon = 0.001, maxStep = 10):
    #     # Max number of iteration
    #     self.maxStep = maxStep
    #     # convergence criteria
    #     self.epsilon = epsilon 
    #     # Number of possible states
    #     self.S = T.shape[0]
    #     # Number of possible observations
    #     self.O = E.shape[0]
    #     self.prob_state_1 = []
    #     # Emission probability
    #     self.E = torch.tensor(E)
    #     # Transition matrix
    #     self.T = torch.tensor(T)
    #     # Initial state vector
    #     self.T0 = torch.tensor(T0)

    
    def __init__(self, path, e_path=None, T=10):
        data = torch.load(path)
        self.T = data['transition']
        self.mask = data['mask']
        self.T0 = data['pi']
        self.S = self.T.shape[0]
        obs = data['obs']
        if e_path is None:
            self.E = torch.zeros((obs.shape[0], self.S))
            for i in tqdm(range(T)):
                y = obs[i]
                densities = evaluate_GMM(data['int2id'], data['id2int'], y, data['allmeans'], data['allvariances'], data['weights'], mask=self.mask, id2state=data['id2state'])
                self.E[i] = densities
            torch.save(self.E, 'emission.pt')
        else:
            self.E = torch.load(e_path)
            self.E = self.E.type(torch.float64)
        self.O = obs.shape[0]


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
        self._backward(obs_prob_seq_rev)