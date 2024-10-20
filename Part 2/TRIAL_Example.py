import argparse
import smartQAOA as sQAOA #import the standard QAOA
import smartMQAOA as smQAOA #import the pretrained QAOA
import numpy as np
import networkx as nx
import time

################# QAOA EXAMPLE ############################

# both pretrained and standard QAOA runs

#use flags to set the number of qubits, depth of the circuit and number of iterations
#recomended to set max iters~10 for a runtime of around 1 hour. Note to run
#a depth other than 256 and 16 the pretraining must first be completed and an approporiate file
# with initial parameters created

#the run time scales as roughly 3mins~1 iteration on a personal deviceTo achieve
#better results this hsould be set to around 500 (will usually only take ~250)

def to_int(x):
    # Reverse the order of the bits to convert back to normal binary representation
    return int(''.join([str(i) for i in x[::-1]]), 2)

def to_binary(x, bits):
    # Convert the integer to binary, fill with leading zeros, and reverse the order
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)][::-1])

# RUN THE STANDARD QAOA CIRCUIT WITHOUT PRETRAINING
def normal(G, n, p, iters=100, times=False, gammas=False):
    Q, qc, gamma_parameters, time_parameters = sQAOA.set_up_sim(n, p, G)
    # can accept parameters if uyou wish to train in chunks of iterations
    if type(times) == bool:
        times = np.random.normal(1., 0.5, size=(p,))
    else:
        times = np.array(times)
    if type(gammas) == bool:
        gammas = np.random.normal(1., 0.5, size=(p,))
    else:
        gammas = np.array(gammas)
    print("Maxcut: ", max(Q), " Average: ", np.mean(Q), " Median: ", np.median(Q))
    result1 = sQAOA.expectation(Q, qc, G, gamma_parameters, time_parameters, gammas, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters})
    print(result1)
    print("ExpectationNormal: ", result1[-2], " after ", result1[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result1[-1])
    return result1,max(Q),np.median(Q)


#RUN THE HYBRID QAOA CIRCUIT WITH PRETRAINING
def pretrained_QAOA(G, GP, params_init, n, p, iters=100, gammaG=False):
    Q, qc, gammaG_parameters, gammaP_parameters, time_parameters = smQAOA.set_up_sim(n, p, G, GP)
    print("Maxcut: ", max(Q), " Average: ", np.mean(Q), " Median: ", np.median(Q))
    if type(gammaG) == bool:
        gammasG = [0] * p
    else:
        gammasG = np.array(gammaG)
    gammasP = params_init[0]
    times = params_init[1]
    result2 = smQAOA.expectation(Q, qc, G, gammaG_parameters, gammaP_parameters, time_parameters, gammasG, gammasP, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters})
    print("ExpectationGP: ", result2[-2], " after ", result2[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result2[-1])
    return result2,max(Q),np.median(Q)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run QAOA simulations with specified parameters.')
    parser.add_argument('--n', type=int, default=16, help='Number of qubits')
    parser.add_argument('--p', type=int, default=256, help='Depth of the circuit')
    parser.add_argument('--iters', type=int, default=500, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()

    n = args.n
    p = args.p
    iters = args.iters
    s = args.seed
    print("Running ER30 graphs with edge probability: 30%")

    #Import the pretrained parameters
    loaded = np.load(f'Graph_Parameters{n}_{p}.npz')
    params_init = np.array([np.array(loaded[key]) for key in loaded])


    print("Running graph seeds: ", s)
    #define graphs
    GP = smQAOA.random_graph(n, G_type="path")
    G = smQAOA.random_graph(n, G_type="random", weighted=False, seed=s, p=0.3)

    #run standard QAOA
    then = time.time()
    result2, maxcut,mediancut = normal(G, n, p, iters=iters)
    print(result2)
    print("Time for STANDARD QAOA: ", time.time() - then)

    #run pretrained QAOA
    then = time.time()
    result2, maxcut,mediancut = pretrained_QAOA(G,GP,params_init, n, p, iters=iters)
    print(result2)
    print("Time for PRETRAINED QAOA: ", time.time() - then)
