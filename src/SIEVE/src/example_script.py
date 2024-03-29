from Viterbi import Sieve
import numpy as np 
import random
from memory_profiler import memory_usage

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
        
    B = np.full((n_states,n_observables), float(np.random.rand(1).item()))
    
    B = B/B.sum(axis=1)[:,None]
    
    return B 


if __name__ == '__main__': 
    
        # seed for data generation 
        sd = 12
        # number of observable symbols 
        n_observables = 50
        # number of states 
        K = 20
        random.seed(sd)
        T = 2000
        # vector of observations 
        y = [random.randint(0,n_observables-1) for _ in range(T)] 
        
        # generate simple data 
        A = create_A_b(n_nodes = K, sd = sd)
        B = create_B(n_states = K, sd = sd) 
        
        # uniform initial probabilities 
        pi = np.full(K, 1 / K)
  
        print("Starting Vanilla Viterbi .. \n ") 
        
        vit = Sieve(pi, A, B, y)
        viterbi_path, t1, t2 = vit.viterbi() 
        print(max(memory_usage(vit.viterbi)))
        
        print("Starting Checkpoint Viterbi .. \n ") 
        
        vit = Sieve(pi, A, B, y)
        x_checkpoint = vit.viterbi_checkpoint() 
        print(max(memory_usage(vit.viterbi_checkpoint, interval=0.0000001)))
        
        print("Checkpoint Viterbi done .. \n" + "path: " + str(x_checkpoint)) 
        
        print("Starting Sieve .. \n " + "path edges: ") 
        
        vit = Sieve(pi, A, B, y)
        indices = [x for x in range(K)] 
        pi = np.full(K, 1 / K)
        vit.initial_state = None 
                
        # prepocessing 
        vit.viterbi_preprocessing_descendants_pruning_root(indices, T, K)
        vit.viterbi_preprocessing_ancestors_pruning_root(indices,  T, K)
        vit.sieve(indices, A, B, y, Pi =pi, K=K)
        print(max(memory_usage((vit.sieve, (indices, A, B, y,), {'Pi': pi, 'K':K}))))
        vit.pretty_print_path(vit.path)
        
        print("Sieve done .. \n" ) 
        
        print("Starting Sieve Middle Path.. \n " + "path edges: ") 
        
        vit = Sieve(pi, A, B, y)
        indices = [x for x in range(K)] 
        pi = np.full(K, 1 / K)
        vit.initial_state = None 
              
        vit.sieve_middlepath(indices, A, B, y, Pi =pi, K=K)
        vit.pretty_print_path(vit.mp_path)
        print("Sieve Middlepath done .. \n" ) 
