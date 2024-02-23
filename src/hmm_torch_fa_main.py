import numpy as np
import random
from hmm_torch_fa import HiddenMarkovModel
import torch
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tracemalloc

def measure(model, T):
    # y = np.array([random.randint(0,n_observables-1) for _ in range(T)])
    # print(y)

    # generate simple data 
    # A = create_A_b(n_nodes = K, sd = sd)
    # B = create_B(n_observables=n_observables, n_states = K, sd = sd).transpose()

    # uniform initial probabilities 
    # pi = np.full(K, 1 / K)

    # model = HiddenMarkovModel(A, B, pi)

    model.N = T
    # model.initialize_forw_back_variables([model.N, model.S])
    obs_prob_seq = model.E[:T]

    tracemalloc.start()
    _, time, mem = model.forward_backward(obs_prob_seq)
    posterior = model.forward * model.backward
    mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # marginal per timestep
    marginal = torch.sum(posterior, 1)
            
    # Normalize porsterior into probabilities
    posterior = posterior / marginal.view(-1, 1)
    print(posterior[-1])

    return time, mem[1]
    
def visualization(data, x, y, title, fname):
    plt.figure()
    g2 = sns.lineplot(data=df, x=x, y=y)
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

hmm = HiddenMarkovModel('./forced_alignment_hmm.pt', './emission.pt')
rows = []

for t in tqdm(range(1, 10, 1)):
    time, mem = measure(hmm, t)
    rows.append({'no obs': t, 'mem': mem, 'time': time})
df = pd.DataFrame(rows)

visualization(df, 'no obs', 'mem', 'mem wrt no. obs', 'mem_obs_fa.pdf')
visualization(df, 'no obs', 'time', 'time wrt no. obs', 'time_obs_fa.pdf')
# g2 = sns.lineplot(data=df, x='no states', y='time')
# g2.set_title('Time plot with No. obs')
# g2.figure.savefig('time_plot_2.pdf', bbox_inches='tight')