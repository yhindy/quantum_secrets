import numpy as np
from typing import List


from pyquil import Program
from pyquil.gates import MEASURE, I, CNOT, CZ, X, Z, H, RZ, PHASE, CPHASE, MEASURE, RX
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection
from pyquil.api import WavefunctionSimulator
from pyquil.quil import DefGate


sim = WavefunctionSimulator(random_seed=1337)
qvm = QVMConnection(random_seed=1337)

def orig_encode(qubit):
    qubit_a = 4
    qubit_b = 3
    qubit_c = 1
    qubit_d = 0

    # qubit_a = QubitPlaceholder()
    # qubit_b = QubitPlaceholder()
    # qubit_c = QubitPlaceholder()
    # qubit_d = QubitPlaceholder()
    
    pi_rot = np.array([[-1.0, 0], [0, -1.0]])
    pi_rot_def = DefGate("PI-ROT", pi_rot)
    PI_ROT = pi_rot_def.get_constructor()
    
    code_register = [qubit_a, qubit_b, qubit, qubit_c, qubit_d]

    pq = Program(H(qubit_a), H(qubit_b), H(qubit_c))
    pq += pi_rot_def
    pq += PI_ROT(qubit_d).controlled(qubit_b).controlled(qubit).controlled(qubit_c)
    pq += [X(qubit_b), X(qubit_c)]
    pq += PI_ROT(qubit_d).controlled(qubit_b).controlled(qubit).controlled(qubit_c)
    pq += [X(qubit_b), X(qubit_c)]
    pq += CNOT(qubit, qubit_d)
    pq += CNOT(qubit_a, qubit)
    pq += CNOT(qubit_a, qubit_d)
    pq += CNOT(qubit_c, qubit)
    pq += CNOT(qubit_b, qubit_d)
    pq += PI_ROT(qubit).controlled(qubit_c).controlled(qubit_d)
  
    return pq, code_register

def orig_decode(qubits, indices=None, measure=True):

    ## indices are qubits you recover secret from, need 3
    ## measure asks whether or not to measure result at end
    if indices is None:
        indices = np.random.choice(len(qubits), 3, replace=False)
    new_qubits = [qubits[index] for index in indices]
    pq = Program()
    ro = pq.declare('ro', 'BIT', 3)
    
    # first hadamard qubit 0
    pq += H(new_qubits[0])
    
    # bell state measurement on 1,2
    pq += X(new_qubits[2])
    pq += CNOT(new_qubits[2], new_qubits[1])
    pq += H(new_qubits[2])
    pq += MEASURE(new_qubits[1], ro[0])
    pq += MEASURE(new_qubits[2], ro[1])
    case1 = Program()
    case2 = Program()
    case1.if_then(ro[1], Program(Z(new_qubits[0]), X(new_qubits[0])), Z(new_qubits[0]))
    case2.if_then(ro[1], X(new_qubits[0]), I(new_qubits[0]))
    pq.if_then(ro[0], case1, case2)
    if measure:
    	pq += MEASURE(new_qubits[0], ro[2])
    return pq, new_qubits


def NN_encode(qubit: QubitPlaceholder, N: int) -> (Program, List[QubitPlaceholder]):
    
    ### qubit: qubit you want to encode (main qubit)
    ### N: number of qubits you want to encode the main qubit in. 
    ### For N=1, there is no encoding
    
    code_register = QubitPlaceholder.register(N)  # the List[QubitPlaceholder] of the qubits you have encoded into
    code_register[0] = qubit
    
    pq = Program()
    
    ### creation of GHZ state: 
    for ii in range(N-1):
        pq += CNOT(code_register[ii],code_register[ii+1])
        
        
    for jj in range(N-1):
        pq += H(code_register[jj])
        pq += CZ(code_register[jj+1],code_register[jj])
        
    
    for kk in range(N-1):
        pq += CNOT(code_register[kk],code_register[-1])

    return pq, code_register


def NN_decode(code_register: List[QubitPlaceholder]) -> (Program, QubitPlaceholder):
    
    out = QubitPlaceholder()
    n = len(code_register)
    
    pq = Program()
    
    ### This is where the magic happens
    for ii in range(n):
        pq += CNOT(code_register[ii],out)
        
        
    return pq, out


def NN_test(qubit, prep_program, N_encode, N_decode):
    
    ### qubit: qubit we want to encode (main qubit)
    ### prep_program: arbitrary program to put the main qubit in the state we want to 
    ### transmit
    ### N_encode: number of qubits to encode the main qubit in
    ### N_decode: number of qubits to read out
    ### if N_decode < N_encode, we will always get alpha squared = beta squared = 0.5,
    ### no matter what we originally encoded the qubit in.
    ### Note that this test only gives the absolute squared values of alpha and beta

    pq = Program()
    pq += prep_program
        
    prog, code_register = NN_encode(qubit,N_encode)

    pq += prog

    progg, out = NN_decode(code_register[0:N_decode])

    pq += progg

    ro = pq.declare('ro', 'BIT', 1)
    pq += MEASURE(out, ro)

    result = np.sum(qvm.run(address_qubits(pq), trials=1000))/1000.0
    
    alpha_sqrd = 1.0 - result
    beta_sqrd = result
    
    print('alpha squared = {}'.format(alpha_sqrd))
    print('beta squared = {}'.format(beta_sqrd))
    
    return alpha_sqrd, beta_sqrd