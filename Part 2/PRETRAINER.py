import smartQAOA as QAOA
import smartMQAOA as mQAOA
import numpy as np
import networkx as nx
import time
import argparse



n=16
p=256
loaded = np.load(f'Parameters{n}_{p}.npz')
params_init = np.array([np.array(loaded[key]) for key in loaded])
#print(params_init)
GP = QAOA.random_graph(n,G_type="path")
Q,qc,gamma_parameters,time_parameters = QAOA.set_up_sim(n,p,GP)
gammas = params_init[0]
times = params_init[1]
result = QAOA.expectation(Q,qc,GP, gamma_parameters, time_parameters, gammas, times, p,optimize=False,return_max_prob=True)
print(result)

print("First set of Gammas: ",gammas[:10])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run QAOA simulations with specified parameters.')
    parser.add_argument('--trial', type=int, default=150, help='Specifies trial set name for file')

    args = parser.parse_args()

    trial = args.trial
    print("pretraining QAOA")
    then =time.time()
    np.random.seed(None)
    gammas = np.random.uniform(0.0,2*np.pi, size=(p,))
    times = np.random.uniform(0.0,2*np.pi ,size=(p,))
    print("Maxcut: ", max(Q)," Average: ",np.mean(Q)," Median: ",np.median(Q))  
    result0=QAOA.expectation(Q,qc,GP, gamma_parameters, time_parameters, gammas, times, p,optimize=True,options={"maxfun":50000,"maxiter":500})
    print("ExpectationNormal: ", result0[-2][-1], " after ", result0[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result0[-1])
    print(result0[0][:10])
    print("time1: ",time.time()-then)

    np.savez(f'Graph_Parameters{n}_{p}_trial_{trial}', *result0[:2])