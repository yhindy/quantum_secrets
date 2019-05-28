import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, I, CNOT, X, Z, H, RZ, PHASE, CPHASE, MEASURE, RX
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection
from pyquil.api import WavefunctionSimulator
from pyquil.quil import DefGate


sim = WavefunctionSimulator(random_seed=1337)
qvm = QVMConnection(random_seed=1337)

def encode(qubit):
    qubit_a = 4
    qubit_b = 3
    qubit_c = 1
    qubit_d = 0
    
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

def decode(qubits, indices=None, measure=True):
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