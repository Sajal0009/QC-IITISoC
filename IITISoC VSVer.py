import jax
import pennylane as qml
from jax import numpy as np

symbols, coordinates = qml.qchem.read_structure('H2O.xyz')
H, NoOfQ = qchem.molecular_hamiltonian(symbols, coordinates, active_electrons = 8, active_orbitals = 6)    #The active electrons and orbitals are part of
                                                                                                           #the process to 'optimise' the molecular 
                                                                                                           #geometry of water; only these electrons
                                                                                                           #are excited/de-excited to obtain Hamiltonian.
                                                                                                           #This increases efficiency of the process wrt
                                                                                                           #using also the inactive s-e of O.

print("Number of qubits required to perform quantum simulations: {:}".format(NoOfQ))
print(H)

dev = qml.device("lightning.qubit", wires = NoOfQ)

hf = qml.qchem.hf_state(8, NoOfQ)
print(hf)                                                                                                  #Printing the Hartree-Fock State

SingleE, DoubleE = qml.qchem.excitations(8, 12)                                                            #Checking all excitations

for excitation in SingleE:
    print(excitation)
for excitation in DoubleE:
    print(excitation)

@qml.qnode(dev)

def Energy(state):
     qml.BasisState(np.array(state), wires = range(NoOfQ))
     return qml.expval(H)

print(Energy(hf))

i = 0
j = 0
def circuit(theta, wires):
    qml.BasisState(hf, wires = wires)
    #for k in range(len(SingleE)):
        #qml.SingleExcitation(theta, wires = SingleE[i])
    #for j in range(len(DoubleE)):
        #qml.DoubleExcitation(theta, wires = DoubleE[j])
    qml.DoubleExcitation(theta, wires = [0, 1, 9, 10])
    return qml.expval(H)
    return qml.state()

print(circuit(np.array(0.), wires = range(NoOfQ)))
def CostFn(theta):
    return circuit(theta, wires = range(NoOfQ))

#print(CostFn(np.array(0.)))

import optax
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)
MaxIt = 100                                                    #The maximum number of iterations of the gradient descent
Tol = 1e-7                                                   #The tolerance, i.e., the maximum value of the difference between consecutive expvals
                                                               #so that it can be considered that value itself
opt = optax.sgd(learning_rate = 0.4)                           #(no clue)

VarAngle = np.array(0.)                                        #The variable angle that changes (rotational phi)

EnergyList = [CostFn(VarAngle)]                                #List of energies returned
print("Current energy:", EnergyList[-1])
AngleList = [VarAngle]                                         #List of rotational angles
State = opt.init(VarAngle)                                       

for n in range(MaxIt):
    Gradient = jax.grad(CostFn)(VarAngle)                      #Gradient of the cost function
    update, State = opt.update(Gradient, State)                #Next two lines both updating the Gradient and the VarAngle depending
    VarAngle = optax.apply_updates(VarAngle, update)           #on the gradient just calculated

    AngleList.append(VarAngle)                                 #Adding the new angle to the end of the angle list
    EnergyList.append(CostFn(VarAngle))                        #Adding the new energy to the end of the energy list
    
    print(EnergyList[-1], AngleList[-1])
    if n > 0:
        Convergence = np.abs(EnergyList[-1] - EnergyList[-2])
        print(Convergence)
        if (Convergence < Tol):
            break

print("The final energy:")
print(EnergyList[-1])
