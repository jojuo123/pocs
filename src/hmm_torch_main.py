import numpy as np
import random
from hmm_torch import HiddenMarkovModel
import torch
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tracemalloc

def create_A_b(n_nodes = 100, sd = 1, prob=0.2): 
    
    np.random.seed(sd) 
    matrix = np.zeros((n_nodes,n_nodes),dtype=float) 
    allstates = [x for x in range(n_nodes)]
    
    for state in range(n_nodes): 
        
        edge_per_node=np.random.binomial(n_nodes-1, p=prob, size=None)
        
        #we sample @edge_per_node edges to connect to current state 
        state_connections = np.random.choice(allstates, size=edge_per_node)
        
        #sample probabilities  
        ps = np.random.uniform(0.01,1, size=edge_per_node)
        
        for i in range(edge_per_node): 
            connection = state_connections[i]
            p = ps[i]
            matrix[state][connection] = p
    
    # normalize matrix 
    for i in range(n_nodes): 
        s = np.sum(matrix[i,]) 
        matrix[i,] = matrix[i,] / sum(matrix[i,]) 
       
    return matrix             
                

def create_B(n_observables = 100, n_states = 100, sd = 1): 
    
    ''' create matrix of uniform emission probabilities '''
    
    np.random.seed(sd) 
        
    B = np.full((n_states,n_observables), float(np.random.rand(1)))
    
    B = B/B.sum(axis=1)[:,None]
    
    return B 

def dptable(state_prob, states):
    print(" ".join(("%8d" % i) for i in range(state_prob.shape[0])))
    for i, prob in enumerate(state_prob.T):
        print("%.7s: " % states[i] +" ".join("%.7s" % ("%f" % p) for p in prob))

sd = 12
# number of observable symbols 
n_observables = 50
# number of states 
K = 100
random.seed(sd)
T = 14
# vector of observations 

def measure(n_observables, K, T):
    y = np.array([random.randint(0,n_observables-1) for _ in range(T)])
    # print(y)

    # generate simple data 
    A = create_A_b(n_nodes = K, sd = sd)
    B = create_B(n_observables=n_observables, n_states = K, sd = sd).transpose()

    # uniform initial probabilities 
    pi = np.full(K, 1 / K)

    model = HiddenMarkovModel(A, B, pi, step=T//2)

    model.N = y.shape[0]
    # model.initialize_forw_back_variables([model.N, model.S])
    obs_prob_seq = model.E[y]

    tracemalloc.start()
    _, time, mem = model.forward_backward(obs_prob_seq)
    posterior = model.forward * model.backward
    mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # print(model.forward[-1])

    # marginal per timestep
    marginal = torch.sum(posterior, 1)
            
    # Normalize porsterior into probabilities
    posterior = posterior / marginal.view(-1, 1)

    return time, mem[1]

def measurec(n_observables, K, T):
    y = np.array([random.randint(0,n_observables-1) for _ in range(T)])
    # print(y)

    # generate simple data 
    A = create_A_b(n_nodes = K, sd = sd)
    B = create_B(n_observables=n_observables, n_states = K, sd = sd).transpose()

    # uniform initial probabilities 
    pi = np.full(K, 1 / K)

    n_checkpoint = 2

    model = HiddenMarkovModel(A, B, pi, step=T//n_checkpoint)

    model.N = y.shape[0]
    # model.initialize_forw_back_variables([model.N, model.S])
    obs_prob_seq = model.E[y]

    tracemalloc.start()
    _, timec, memc = model.forward_backward_checkpoint(obs_prob_seq)
    # posterior = model.forward * model.backward
    memc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # print(model.forward[-1])

    # marginal per timestep
    # marginal = torch.sum(posterior, 1)
            
    # Normalize porsterior into probabilities
    # posterior = posterior / marginal.view(-1, 1)

    return timec, memc[1]

# results = [model.forward.cpu().numpy(), model.backward.cpu().numpy(), posterior.cpu().numpy()]
# result_list = ["Forward", "Backward", "Posterior"]

# for state_prob, path in zip(results, result_list) :
#     inferred_states = np.argmax(state_prob, axis=1)
#     print()
#     print(path)
#     dptable(state_prob, states={i: i for i in range(model.S)})
#     print()

# print(measure(n_observables, K, T))

def visualization(data, x, y, title, fname):
    plt.figure()
    g2 = sns.lineplot(data=df, x=x, y=y, hue='func')
    g2.set_title(title)
    g2.figure.savefig(fname, bbox_inches='tight')

# rows = []

# for k in tqdm(range(10, 1001, 10)):
#     time, mem = measure(n_observables, k, T)
#     rows.append({'no states': k, 'mem': mem, 'time': time})

# df = pd.DataFrame(rows)
# # print(df)
# # g = sns.lineplot(data=df, x='no states', y='mem')
# # g.set_title('Mem plot with No. states')
# # g.figure.savefig('mem_plot.pdf', bbox_inches='tight')
# visualization(df, 'no states', 'mem', 'mem wrt no. states', 'mem_state.pdf')
# visualization(df, 'no states', 'time', 'time wrt no. states', 'time_state.pdf')
# g2 = sns.lineplot(data=df, x='no states', y='time')
# g2.set_title('Time plot with No. states')
# g2.figure.savefig('time_plot.pdf', bbox_inches='tight')

rows = []

for t in tqdm(range(10, 5001, 10)):
    time, mem= measure(n_observables, K, t)
    rows.append({'no obs': t, 'mem': mem, 'time': time, 'func': 'normal'})
    timec, memc = measurec(n_observables, K, t)
    rows.append({'no obs': t, 'mem': memc, 'time': timec, 'func': 'checkpoint'})
df = pd.DataFrame(rows)



visualization(df, 'no obs', 'mem', 'mem wrt no. obs', 'mem_obs_new.pdf')
visualization(df, 'no obs', 'time', 'time wrt no. obs', 'time_obs_new.pdf')
# g2 = sns.lineplot(data=df, x='no states', y='time')
# g2.set_title('Time plot with No. obs')
# g2.figure.savefig('time_plot_2.pdf', bbox_inches='tight')