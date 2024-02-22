import forward_backward_hmm

p = forward_backward_hmm.load_probabilities("prob_vector.pickle")
h = forward_backward_hmm.HMM(p)
print(h)
s = 'a'
print(h.forward_probability(h.forward(s)))
