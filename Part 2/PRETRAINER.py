import smartQAOA as QAOA
import smartMQAOA as mQAOA
import numpy as np
import networkx as nx
import time
import argparse

################# Pretraining QAOA EXAMPLE ############################

# max iterations is set to 10 to have runtime of around 10 minutes. To achieve 
# better results set max iters to 500 (usually only needs ~200)

n=16
p=256
loaded = np.load(f'Graph_Parameters{n}_{p}.npz')
params_init = np.array([np.array(loaded[key]) for key in loaded])
#print(params_init)
GP = QAOA.random_graph(n,G_type="path")
Q,qc,gamma_parameters,time_parameters = QAOA.set_up_sim(n,p,GP)
gammas = params_init[0]
times = params_init[1]
result = QAOA.expectation(Q,qc,GP, gamma_parameters, time_parameters, gammas, times, p,optimize=False,return_max_prob=True)
print(result)
print("Pretrained parameters give an approximation ratio for the path graph of: ",result[0]/15, " with probibility of getting correct partition: ",result[1])
# these are the params used for primary trials in thesis
print("First set of Gammas: ",gammas[:10])



print("pretraining QAOA")
then =time.time()
np.random.seed(None)
gammas = np.random.uniform(0.0,2*np.pi, size=(p,))
times = np.random.uniform(0.0,2*np.pi ,size=(p,))
print("Maxcut: ", max(Q)," Average: ",np.mean(Q)," Median: ",np.median(Q))  
result0=QAOA.expectation(Q,qc,GP, gamma_parameters, time_parameters, gammas, times, p,optimize=True,options={"maxfun":50000,"maxiter":10})
print("ExpectationNormal: ", result0[-2], " after ", result0[-3].nit, " iterations gives a prob of measuring MAXCUT of: ", result0[-1])
print(result0[0][:10])
print("time1: ",time.time()-then)
