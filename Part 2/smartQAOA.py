import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter

# returns binary arrays for each of the input integers in x
def unpackbits(x, num_bits, asbool=True):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits-1,-1,-1, dtype=x.dtype).reshape([1, num_bits])
    if asbool:
        return (x & mask).astype(bool).reshape(xshape + [num_bits])
    else:
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

'''_____________________________________ Generating Random Weighted Graph ____________________________________________'''

# The following code generates a random graph for maxcut
# The erdos-renyi model is used, in which each possible edge is included with some probability (in this case p)

def random_graph(n, p=0.5, seed=42, weighted=False, G_type="random"):
    if type(seed) == int:
        np.random.seed(seed)
    else:
        if seed == True:
            np.random.seed(42)
    
    if G_type == "path":
        G = np.zeros((n, n))
        for i in range(n - 1):
            if weighted:
                G[i, i + 1] = np.random.random()
            else:
                G[i, i + 1] = 1
            G[i + 1, i] = G[i, i + 1]
    if G_type == "cycle":
        G = np.zeros((n, n))
        for i in range(n-1):
            if weighted:
                G[i, i + 1] = np.random.random()
            else:
                G[i, i + 1] = 1
            G[i + 1, i] = G[i, i + 1]
        if weighted:
            G[n-1, 0] = np.random.random()
        else:
            print("here")
            G[n-1, 0] = 1
            G[0, n-1] = 1
    if G_type == "random":
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < p:
                    if weighted:
                        G[i, j] = np.random.random()
                    else:
                        G[i, j] = 1
                    G[j, i] = G[i, j]
    if G_type == "complete":
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if weighted:
                    G[i, j] = np.random.random()
                else:
                    G[i, j] = 1
                G[j, i] = G[i, j]                    
    
    return G


def convert_qubo(G):
    return -(G - np.diag(G.sum(axis=1)))

'''_____________________________  Computing quality vector _____________________________'''

def get_cut(soln,Q):
    return soln.T @ Q @ soln

# Note the quality vector is symmetric about centre point
def get_quality_vector(G):
    n = len(G)
    Q = convert_qubo(G)
    N = 2**n
    quals = np.zeros(N)
    all_binary_arrays = unpackbits(np.arange(N//2), n, asbool=True)
    for i in range(N//2):
        quals[i] = get_cut(all_binary_arrays[i],Q)
    quals[N//2:] = (quals[:N//2])[::-1]
    return quals

def initialize_qubits(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return qc

def apply_mixing_operator(qc, t):
    for qubit in range(qc.num_qubits):
        qc.rx(2*t, qubit)

def apply_phase_separator(qc, G, gamma):
    n = qc.num_qubits
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(-gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)

def reverse_mixing_operator(n,t):
    qc = QuantumCircuit(n)
    for qubit in range(qc.num_qubits):
        qc.rx(-2*t, qubit)
    return qc

def reverse_phase_separator(n,G, gamma):
    qc = QuantumCircuit(n)
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)
    return qc

def map_params(gammas, times, qc, gamma_parameters, time_parameters, p):
    params_dict = {}
    for i in range(p):
        params_dict[gamma_parameters[i]] = gammas[i]
        params_dict[time_parameters[i]] = times[i]
    qc = qc.assign_parameters(params_dict)
    return qc

def amplified_state(qc, gamma_parameters, time_parameters, gammas, times, p):
    qc = map_params(gammas, times, qc, gamma_parameters, time_parameters, p)
    backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
    return np.abs(backend.run(qc).result().get_statevector())**2



def expectation(Q, qc,G, gamma_parameters, time_parameters, gammas, times, p, return_max_prob=False,optimize=False, method='L-BFGS-B', bounds=None,options = {'maxiter': 100} ):
    n=qc.num_qubits
    def cost(params):
        # Unpack the parameters
        gammas_opt = params[:p]
        times_opt = params[p:]
        # Map parameters to quantum circuit
        qc_opt = map_params(gammas_opt, times_opt, qc, gamma_parameters, time_parameters, p)
        # Use the backend to calculate the expectation value
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        statevector = backend.run(qc_opt).result().get_statevector()
        
        expectation_value = Q @ (np.abs(statevector)**2)
        return -expectation_value  # Negative for minimization
    derivativeM = []
    for i in range(n):
        dM=QuantumCircuit(n)
        dM.x(i)
        derivativeM+= [dM]

    # gradients are calculated according to the procedure outlined by Jones et al 2024, reference 144 of the associated thesis
    def gradients(params):
        # Unpack the parameters
        gammas_opt = params[:p]
        times_opt = params[p:]
        # Initialize the statevector
        # Map parameters to quantum circuit
        qc_opt = map_params(gammas_opt, times_opt, qc, gamma_parameters, time_parameters, p)
        # Use the backend to calculate the expectation value
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        sv= backend.run(qc_opt).result().get_statevector()
        phi = Statevector(sv.data)
        lm=sv.data*Q
        lam = Statevector(lm)
        # compute gradients
        dTimes = np.zeros(p)
        dGammas = np.zeros(p)   
        for i in range(p-1,-1,-1):

            dTimes[i] = 2*sum(1.0j*lam.conjugate().data@phi.evolve(derivativeM[i]).data for i in range(n)).real

            udag = reverse_mixing_operator(n,times_opt[i])
            phi=phi.evolve(udag)

            lam=lam.evolve(udag)

            dGammas[i] = 2*(1j*lam.conjugate().data@(Q*phi.data)).real

            udag = reverse_phase_separator(n,G,gammas_opt[i])
            phi=phi.evolve(udag)

            lam=lam.evolve(udag)
        return np.concatenate((dGammas, dTimes))

    if optimize:
        # Combine initial gamma and time parameters into one array
        initial_params = np.concatenate((gammas, times))
        
        # Default bounds if not provided
        if bounds is None:
            bounds = [(0, 2*np.pi)] * (2*p)  # Assuming ranges [0, 2π] for all parameters
        # Minimize the negative expectation value
        result = minimize(cost, initial_params, bounds=bounds, method=method,jac=gradients,options=options)
        
        # Extract the optimal parameter
        optimal_params = result.x
        optimal_gammas = optimal_params[:p]
        optimal_times = optimal_params[p:]
        # Get the probablility of measuring the best value

        qc = map_params(optimal_gammas, optimal_times, qc, gamma_parameters, time_parameters, p)
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        
        # The maximum expectation value is the negative of the minimum found
        max_expectation_value = -result.fun
        
        return optimal_gammas, optimal_times, result,max_expectation_value,sum(probs[np.where(Q == max(Q))[0]])
    
    else:
        
        # Calculate expectation with given parameters
        qc = map_params(gammas, times, qc, gamma_parameters, time_parameters, p)
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        if return_max_prob:
            return Q @ probs, sum(probs[np.where(Q == max(Q))[0]]),probs
        else:
            return Q @ probs



def set_up_sim(n,p,G):
    Q = get_quality_vector(G)
    qc = initialize_qubits(n)
    gamma_parameters = [Parameter(f"gamma_{i}") for i in range(p)]
    time_parameters = [Parameter(f"t_{i}") for i in range(p)]
    for i in range(p):
        apply_phase_separator(qc, G, gamma_parameters[i])
        apply_mixing_operator(qc, time_parameters[i])
    qc.save_statevector()
    qc = transpile(qc, backend=AerSimulator())
    return Q,qc,gamma_parameters,time_parameters

