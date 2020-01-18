from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer


if __name__ == "__main__":
    
    # define Deutsch Oracle
    def isConst(func, circuit, qreg, creg):
        circ = circuit
        q = qreg
        c = creg
        
        # preprocessing
        circ.x(q[0])
        circ.x(q[1])
        circ.h(q[0])
        circ.h(q[1])
        
        # run through blackbox
        func(circ, q)
        
        # remove superposition
        circ.h(q[0])
        circ.h(q[1])
        
        # measure
        circ.measure(q, c)
        
        # run circuit
        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(circ, backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circ)
        print(counts)
        
        
    def blackbox_const0(circuit, qreg):
        return None
    
    
    def blackbox_const1(circuit, qreg):
        circ = circuit
        q = qreg
        circ.x(q[0])
        return None
    
    
    def blackbox_identity(circuit, qreg):
        circ = circuit
        q = qreg
        circ.cx(q[1], q[0])
        return None
    
    
    def blackbox_negate(circuit, qreg):
        circ = circuit
        q = qreg
        circ.cx(q[1], q[0])
        circ.x(q[0])
        return None
        
    # Create a quantum register w/ 2 qubits and corresponding classical reg
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    
    # Create a quantum circuit acting on the q register
    circ = QuantumCircuit(q,c)
    
    # Run oracle
    isConst(blackbox_const1, circ, q, c)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    