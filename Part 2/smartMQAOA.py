import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
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

# The following code generates the graph structures for maxcut
# The for random graphs the erdos-renyi model is used, in which each possible edge is included with some probability (in this case p)


def random_graph(n, p=0.5, seed=42, weighted=False, G_type="random"):
    """
    Generates different types of graphs based on the specified parameters.

    Parameters:
    - n (int): Number of nodes.
    - p (float): Probability for edge creation in random graphs.
    - seed (int or bool): Seed for random number generator.
    - weighted (bool): Whether to assign random weights to edges.
    - G_type (str): Type of graph to generate. Options include:
        - "random": Erdos-Renyi random graph.
        - "path": A path graph.
        - "cycle": A cycle graph.
        - "complete": A complete graph (unweighted only).
        - "3-regular": A 3-regular graph.

    Returns:
    - G (np.ndarray): Adjacency matrix of the generated graph.
    """
    # Set the random seed for reproducibility
    if isinstance(seed, int):
        np.random.seed(seed)
    elif seed is True:
        np.random.seed(42)

    if G_type == "path":
        G = np.zeros((n, n))
        for i in range(n - 1):
            G[i, i + 1] = np.random.random() if weighted else 1
            G[i + 1, i] = G[i, i + 1]

    elif G_type == "cycle":
        G = np.zeros((n, n))
        for i in range(n - 1):
            G[i, i + 1] = np.random.random() if weighted else 1
            G[i + 1, i] = G[i, i + 1]
        G[n-1, 0] = np.random.random() if weighted else 1
        G[0, n-1] = G[n-1, 0]

    elif G_type == "random":
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < p:
                    G[i, j] = np.random.random() if weighted else 1
                    G[j, i] = G[i, j]

    elif G_type == "complete":
        if weighted:
            raise ValueError("Complete graphs cannot be weighted. Please set weighted=False.")
        # Create a complete graph without self-loops
        G = np.ones((n, n)) - np.eye(n)
        G = G.astype(int)  # Ensure the adjacency matrix is of integer type

    elif G_type == "3-regular":
        if n * 3 % 2 != 0:
            raise ValueError("3-regular graphs are only possible for an even number of nodes.")
        G_nx = nx.random_regular_graph(3, n, seed=seed if isinstance(seed, int) else None)
        G = np.array(nx.to_numpy_array(G_nx))
        if weighted:
            # Assign random weights to existing edges
            weight_matrix = np.random.random((n, n))
            G = G * weight_matrix

    else:
        raise ValueError(f"Unknown graph type: {G_type}")

    return G




def convert_qubo(G):
    return -(G - np.diag(G.sum(axis=1)))

'''_____________________________  Computing quality vector _____________________________'''

def get_cut(soln,Q):
    return soln.T @ Q @ soln

# Note the quality vector is a mirror image around centre point
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

def apply_phase_separatorG(qc, G, gamma):
    n = qc.num_qubits
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(-gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)
    
def apply_phase_separatorP(qc, G, gamma):
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

def map_params(gammasG, gammasP, times, qc, gammaG_parameters,gammaP_parameters, time_parameters, p):
    params_dict = {}
    for i in range(p):
        params_dict[gammaG_parameters[i]] = gammasG[i]
        params_dict[gammaP_parameters[i]] = gammasP[i]
        params_dict[time_parameters[i]] = times[i]
    qc = qc.assign_parameters(params_dict)
    return qc

def amplified_state(qc, gammaG_parameters,gammaP_parameters, time_parameters, gammasG, gammasP,times, p):
    qc = map_params(gammasG,gammasP, times, qc, gammaG_parameters,gammaP_parameters, time_parameters, p)
    backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
    return np.abs(backend.run(qc).result().get_statevector())**2


# Assuming map_params and other necessary components are already defined

def expectation(Q, qc,G, gammaG_parameters,gammaP_parameters, time_parameters, gammasG,gammasP, times, p, return_max_prob=False,optimize=False, method='L-BFGS-B', bounds=None,options = {'maxiter': 100} ):
    n=qc.num_qubits
    GP = random_graph(n, G_type="path")
    QP = get_quality_vector(GP)
    def cost(params):
        # Unpack the parameters
        gammasG_opt = params[:p]
        gammasP_opt = params[p:2*p]
        times_opt = params[2*p:]
        # Map parameters to quantum circuit
        qc_opt = map_params(gammasG_opt,gammasP_opt, times_opt, qc, gammaG_parameters,gammaP_parameters, time_parameters, p)
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
    def gradients(params):
        # Unpack the parameters
        gammasG_opt = params[:p]
        gammasP_opt = params[p:2*p]
        times_opt = params[2*p:]
        # Map parameters to quantum circuit
        qc_opt = map_params(gammasG_opt,gammasP_opt, times_opt, qc, gammaG_parameters,gammaP_parameters, time_parameters, p)
        # Use the backend to calculate the expectation value
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        sv= backend.run(qc_opt).result().get_statevector()
        phi = Statevector(sv.data)
        lm=sv.data*Q
        lam = Statevector(lm)
        # compute gradients
        dTimes = np.zeros(p)
        dGammasG = np.zeros(p)
        dGammasP = np.zeros(p)   
        for i in range(p-1,-1,-1):
            # compute gradient with respect to t
            dTimes[i] = 2*sum(1.0j*lam.conjugate().data@phi.evolve(derivativeM[i]).data for i in range(n)).real

            # update phi
            udag = reverse_mixing_operator(n,times_opt[i])
            phi=phi.evolve(udag)

            # update lambda
            lam=lam.evolve(udag)

            # compute gradient with respect to gammaP
            dGammasP[i] = 2*(1j*lam.conjugate().data@(QP*phi.data)).real

            # update phi
            udag = reverse_phase_separator(n,GP,gammasP_opt[i])
            phi=phi.evolve(udag)

            # update lambda
            lam=lam.evolve(udag)

            # compute gradient with respect to gammaG
            dGammasG[i] = 2*(1j*lam.conjugate().data@(Q*phi.data)).real

            # update phi
            udag = reverse_phase_separator(n,G,gammasG_opt[i])
            phi=phi.evolve(udag)

            # update lambda
            lam=lam.evolve(udag)
        return np.concatenate((dGammasG, dGammasP, dTimes))

    if optimize:
        # Combine initial gamma and time parameters into one array
        initial_params = np.concatenate((gammasG,gammasP, times))
        
        # Default bounds if not provided
        if bounds is None:
            bounds = [(0, 2*np.pi)] * (3*p)  # Assuming ranges [0, 2π] for all parameters

        costs=[]
        def make_callback(costs):
            iteration = 0
            def callback(xk):
                nonlocal iteration
                iteration += 1
                obj_value = cost(xk)
                costs.append(obj_value)
                print(f"Iteration: {iteration},  Objective function value: {obj_value}")
            return callback

        callback = make_callback(costs)
        # Minimize the negative expectation value
        result = minimize(cost, initial_params, bounds=bounds, method=method,jac=gradients,options=options,callback=callback)
    

        # Extract the optimal parameters
        optimal_params = result.x
        optimal_gammasG = optimal_params[:p]
        optimal_gammasP = optimal_params[p:2*p]
        optimal_times = optimal_params[2*p:]
        
        # Get the probablility of measuring the best value
        qc = map_params(optimal_gammasG, optimal_gammasP,optimal_times, qc, gammaG_parameters,gammaP_parameters, time_parameters, p)
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)

        
        return optimal_gammasG,optimal_gammasP, optimal_times, result, costs,sum(probs[np.where(Q == max(Q))[0]])
    
    else:
        
        # Calculate expectation with given parameters
        qc = map_params(gammasG,gammasP, times, qc, gammaG_parameters,gammaP_parameters, time_parameters, p)
        backend = AerSimulator(method='statevector', device='CPU', cuStateVec_enable=True)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        if return_max_prob:
            return Q @ probs, sum(probs[np.where(Q == max(Q))[0]])
        else:
            return Q @ probs



def set_up_sim(n,p,G,GP):
    Q = get_quality_vector(G)
    qc = initialize_qubits(n)
    gamma_parametersG = [Parameter(f"gammaG_{i}") for i in range(p)]
    gamma_parametersP = [Parameter(f"gammaP_{i}") for i in range(p)]
    time_parameters = [Parameter(f"t_{i}") for i in range(p)]
    for i in range(p):
        apply_phase_separatorG(qc, G, gamma_parametersG[i])
        apply_phase_separatorP(qc, GP, gamma_parametersP[i])
        apply_mixing_operator(qc, time_parameters[i])
    qc.save_statevector()
    qc = transpile(qc, backend=AerSimulator())
    return Q,qc,gamma_parametersG,gamma_parametersP,time_parameters

'''___________________________________________________'''




