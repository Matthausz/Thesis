import argparse
import smartQAOA as sQAOA
import smartMQAOA as smQAOA
import numpy as np
import networkx as nx
import time

def to_int(x):
    # Reverse the order of the bits to convert back to normal binary representation
    return int(''.join([str(i) for i in x[::-1]]), 2)

def to_binary(x, bits):
    # Convert the integer to binary, fill with leading zeros, and reverse the order
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)][::-1])


def pre_train(G, n, p, iters=100):
    Q, qc, gamma_parameters, time_parameters = sQAOA.set_up_sim(n, p, G)
    gammas = np.random.normal(1., 0.5, size=(p,))
    times = np.random.normal(1., 0.5, size=(p,))
    print("Maxcut: ", max(Q), " Average: ", np.mean(Q), " Median: ", np.median(Q))
    result0 = sQAOA.expectation(Q, qc, G, gamma_parameters, time_parameters, gammas, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters,"gtol":1e-8})
    print("ExpectationP: ", result0[-2][-1], " after ", result0[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result0[-1])
    return result0

def normal(G, n, p, iters=100, times=False, gammas=False):
    Q, qc, gamma_parameters, time_parameters = sQAOA.set_up_sim(n, p, G)
    if type(times) == bool:
        times = np.random.normal(1., 0.5, size=(p,))
    else:
        times = np.array(times)
    if type(gammas) == bool:
        gammas = np.random.normal(1., 0.5, size=(p,))
    else:
        gammas = np.array(gammas)
    print("Maxcut: ", max(Q), " Average: ", np.mean(Q), " Median: ", np.median(Q))
    result01 = sQAOA.expectation(Q, qc, G, gamma_parameters, time_parameters, gammas, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters})
    print("ExpectationNormal: ", result01[-2][-1], " after ", result01[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result01[-1])
    return result01,max(Q),np.median(Q)

def trained_hybrid(G, GP, params_init, n, p, iters=100, gammaG=False):
    Q, qc, gammaG_parameters, gammaP_parameters, time_parameters = smQAOA.set_up_sim(n, p, G, GP)
    print("Maxcut: ", max(Q), " Average: ", np.mean(Q), " Median: ", np.median(Q))
    if type(gammaG) == bool:
        gammasG = [0] * p
    else:
        gammasG = np.array(gammaG)
    gammasP = params_init[0]
    times = params_init[1]
    result2 = smQAOA.expectation(Q, qc, G, gammaG_parameters, gammaP_parameters, time_parameters, gammasG, gammasP, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters})
    print("ExpectationGP: ", result2[-2][-1], " after ", result2[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result2[-1])
    return result2,max(Q),np.median(Q)

def random_hybrid(G, GP, n, p, iters=100, times=False):
    Q, qc, gammaG_parameters, gammaP_parameters, time_parameters = smQAOA.set_up_sim(n, p, G, GP)
    gammasG = np.random.normal(1., 0.5, size=(p,))
    gammasP = np.random.normal(1., 0.5, size=(p,))
    if type(times) == bool:
        times = np.random.normal(1., 0.5, size=(p,))
    else:
        print("setting times")
        times = np.array(times)
    result1 = smQAOA.expectation(Q, qc, G, gammaG_parameters, gammaP_parameters, time_parameters, gammasG, gammasP, times, p, optimize=True, options={"maxfun": 50000, "maxiter": iters})
    print("ExpectationG: ", result1[-2][-1], " after ", result1[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result1[-1])
    return result1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run QAOA simulations with specified parameters.')
    parser.add_argument('--n', type=int, default=16, help='Number of qubits')
    parser.add_argument('--p', type=int, default=256, help='Depth of the circuit')
    parser.add_argument('--iters', type=int, default=500, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('-ep','--edge_probability', type=float, default=0.3, help='Random seed')

    args = parser.parse_args()

    n = args.n
    p = args.p
    iters = args.iters
    s = args.seed
    ep= args.edge_probability
    print("Running ER30 graphs with edge probability: ", ep)

    loaded = np.load(f'Parameters{n}_{p}.npz')
    params_init = np.array([np.array(loaded[key]) for key in loaded])

    loaded = np.load(f'Seeds_unweighted_ER30.npz')
    seeds = np.array([np.array(loaded[key]) for key in loaded])
    print("Seed: ", s, "is in the", seeds[s], "percentile. Starting from 0 iterations with full depth")
    
    GP = smQAOA.random_graph(n, G_type="path")
    G = smQAOA.random_graph(n, G_type="random", weighted=False, seed=s, p=ep)

    then = time.time()
    result2, maxcut,mediancut = normal(G, n, p, iters=iters)
    with open(f'FINAL_ER30_RANDOM/R_Seed_{s}_Qubits_{n}_Layers_{p}.txt', 'w') as f:
        f.write(f"{iters} iters for seed {s} on ER{ep} which is in the {seeds[s]} percentile\n Maxcut: {maxcut}, MedianCut: {mediancut}\n")
        for item in result2[-2]:
            f.write(f"{item}\n")
    print(result2)
    print("Time for random hybrid QAOA: ", time.time() - then)
