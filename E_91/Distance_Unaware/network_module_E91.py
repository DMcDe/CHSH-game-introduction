#Import the Node and BSMNode classes, which we will use as the basis for our player's circuits and referees circuit, respectively
from sequence.topology.node import Node, BSMNode

#Import the Memory class, which we will use to hold our qubits
from sequence.components.memory import Memory

#Import the EntanglementGenerationA class, which we will use to entangle our player's qubits
from sequence.entanglement_management.generation import EntanglementGenerationA

#Import the Timeline class, which will allow our simulation to run
from sequence.kernel.timeline import Timeline

#Import the QuantumChannel and ClassicalChannel classes, which allow for communication between Alice and Bob and the referee
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel

#Import the EntanglementProtocol class, which manages the processes of creating entanglement
from sequence.entanglement_management.entanglement_protocol import EntanglementProtocol

#Import the Message class, which enables communication on classical channels
from sequence.message import Message

#Import the Circuit class, which we will use to build custom circuits
from sequence.components.circuit import Circuit, x_gate, y_gate, z_gate, s_gate, t_gate, validator

#Import the Protocol class, which allows us to define custom actions for nodes
from sequence.protocol import Protocol

#Import the QuantumManager class
from sequence.kernel import quantum_manager

#Import the SeQuENCe Logger for Debugging
import sequence.utils.log as log

#Import relevant components from Qutip, a quantum system simulator that SeQUeNCe incorporates
from qutip.qip.circuit import QubitCircuit, CircuitSimulator
from qutip.qip.operations import gate_sequence_product
from qutip import Qobj
import numpy as np
from enum import Enum
import random
import itertools
# --- E91 measurement settings (angles in radians) ---

# Alice's three bases: A0, A1, A2
# Implemented as RY(2 * phi_A) then Z-measurement
E91_ALICE_BASES = np.array([
    0.0,              # A0 = 0°
    np.pi / 4,        # A1 = 45°
    np.pi / 2         # A2 = 90°
])

# Bob's three bases: B0, B1, B2
E91_BOB_BASES = np.array([
    np.pi / 8,        # B0 = 22.5°
    3 * np.pi / 8,    # B1 = 67.5°
    5 * np.pi / 8     # B2 = 112.5°
])

# Map specific (AliceBasis, BobBasis) → CHSH setting (x,y) ∈ {0,1}×{0,1}
# These four pairs are used for Bell/CHSH estimation
E91_CHSH_MAP = {
    (1, 0): (0, 0),   # (A1, B0) → (x=0, y=0)
    (1, 1): (0, 1),   # (A1, B1) → (0, 1)
    (2, 0): (1, 0),   # (A2, B0) → (1, 0)
    (2, 1): (1, 1),   # (A2, B1) → (1, 1)
}

# Basis pair used for key generation (raw E91 key)
E91_KEY_PAIR = (0, 0)  # (A0, B0)

def rx_gate(phi):
    return Qobj([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                     [-1j * np.sin(phi / 2), np.cos(phi / 2)]])

def ry_gate(phi):
    return Qobj([[np.cos(phi / 2), -np.sin(phi / 2)],
                     [np.sin(phi / 2), np.cos(phi / 2)]])


def rz_gate(phi):
    return Qobj([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]])

# U1(α) = [[cos(α), i·sin(α)],
#          [i·sin(α), cos(α)]]
def u1_gate(alpha):
    return Qobj([
        [np.cos(alpha), 1j * np.sin(alpha)],
        [1j * np.sin(alpha), np.cos(alpha)]
    ])

# U2(α) = [[cos(α), -sin(α)],
#          [sin(α),  cos(α)]]
def u2_gate(alpha):
    return Qobj([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])

# U3(α) = [[e^{iα},     0   ],
#          [  0   ,  e^{-iα}]]
def u3_gate(alpha):
    return Qobj([
        [np.exp(1j * alpha), 0],
        [0, np.exp(-1j * alpha)]
    ])

#Define a CustomCircuit class so we can calculate the unitary matrix of our circuits
#The only difference from SeQUeNCe's built-in Circuit class is the addition of our custom gates
class CustomCircuit(Circuit):
    def __init__(self, size: int):
        super().__init__(size)
        
    def get_unitary_matrix(self) -> np.ndarray:
        if self._cache is None:
            if len(self.gates) == 0:
                self._cache = np.identity(2 ** self.size)
                return self._cache

            qc = QubitCircuit(self.size)
            qc.user_gates = {"X": x_gate,
                             "Y": y_gate,
                             "Z": z_gate,
                             "S": s_gate,
                             "T": t_gate,
                             "RX": rx_gate,
                             "RY": ry_gate,
                             "RZ": rz_gate,
                             "U1": u1_gate,
                             "U2": u2_gate,
                             "U3": u3_gate}
            for gate in self.gates:
                name, indices, arg = gate
                if name == 'h':
                    qc.add_gate('SNOT', indices[0])
                elif name == 'x':
                    qc.add_gate('X', indices[0])
                elif name == 'y':
                    qc.add_gate('Y', indices[0])
                elif name == 'z':
                    qc.add_gate('Z', indices[0])
                elif name == 'cx':
                    qc.add_gate('CNOT', controls=indices[0], targets=indices[1])
                elif name == 'ccx':
                    qc.add_gate('TOFFOLI', controls=indices[:2], targets=indices[2])
                elif name == 'swap':
                    qc.add_gate('SWAP', indices)
                elif name == 't':
                    qc.add_gate('T', indices[0])
                elif name == 's':
                    qc.add_gate('S', indices[0])
                elif name == 'phase':
                    qc.add_gate('PHASEGATE', indices[0], arg_value=arg)
                elif name == 'rx':
                    qc.add_gate('RX', indices[0], arg_value=arg)
                elif name == 'ry':
                    qc.add_gate('RY', indices[0], arg_value=arg)
                elif name == 'rz':
                    qc.add_gate('RZ', indices[0], arg_value=arg)
                elif name == 'u1':
                    qc.add_gate('U1', indices[0], arg_value=arg)
                elif name == 'u2':
                    qc.add_gate('U2', indices[0], arg_value=arg)
                elif name == 'u3':
                    qc.add_gate('U3', indices[0], arg_value=arg)
                else:
                    raise NotImplementedError
            self._cache = gate_sequence_product(qc.propagators()).full()

        return self._cache

#Define a manager to update protocols on the player's nodes
class Manager:
    def __init__(self, node, mem_name):
        self.node = node
        self.mem_name = mem_name
        
    def update(self, prot, mem, st):
        if st == 'RAW':
            mem.reset()
    
        if st == 'ENTANGLED':
            if mem.expiration_event is not None:
                mem.timeline.remove_event(mem.expiration_event)
                mem.expiration_event = None

    def add_entanglement_protocol(self, middle: str, other: str):
        self.node.protocols = [EntanglementGenerationA(self.node, '%s.eg' % self.node.name, 
                                                       middle, other, self.node.components[self.mem_name])]
        
    def add_player_protocol(self, rec_name: str, rec_node: str):
        PlayerProtocol(self.node, '%s.pp' % self.node.name, rec_name, rec_node)
    
    def add_alice_protocol(self,theta1,theta2,delta=None,alphas=None):
        AliceProtocol(self.node, '%s.ap' % self.node.name,theta1,theta2,delta,alpha=alphas)
        
    def add_bob_protocol(self,theta1,theta2,delta=None,alphas=None):
        BobProtocol(self.node, '%s.bp' % self.node.name,theta1,theta2,delta,alpha=alphas)

#Define a custom PlayerNode class, with a manager instance variable
class PlayerNode(Node):
    def __init__(self, name: str, tl: Timeline, circ: CustomCircuit, reg: int, fid: int = 1, eff: int = 1, coh: float = -1):
        super().__init__(name, tl)
        
        mem_name = '%s.mem' % name
        mem = Memory(mem_name, tl, fidelity = fid, frequency = 0,
                    efficiency = eff, coherence_time = coh, wavelength = 500)
        mem.owner = self
        mem.add_receiver(self)
        self.add_component(mem)
        self.resource_manager = Manager(self, mem_name)
        self.shared_circ = circ
        assert reg < circ.size, "register can't be bigger than the number of qubits"
        self.reg = reg
                
    def init(self):
        mem = self.get_components_by_type('Memory')[0]
        mem.reset()
        
    def receive_msg(self, src: str, msg: 'Message'):
        self.protocols[0].received_message(src, msg)
    
    def get(self, photon, **kwargs):
        self.send_qubit(kwargs['dst'], photon)

def pair_protocol(node1: Node, node2: Node):
    p1 = node1.protocols[0]
    p2 = node2.protocols[0]
    n1_mem_name = node1.get_components_by_type('Memory')[0].name
    n2_mem_name = node2.get_components_by_type('Memory')[0].name
    p1.set_others(p2.name, node2.name, [n2_mem_name])
    p2.set_others(p1.name, node1.name, [n1_mem_name])
    
# Define a function to add and pair entanglement protocols on both player's nodes
def generate_entanglement(node1: Node, node2: Node, entangler: Node):
    node1.resource_manager.add_entanglement_protocol(entangler.name, node2.name)
    node2.resource_manager.add_entanglement_protocol(entangler.name, node1.name)
    pair_protocol(node1, node2)

#  \Define custom enumerators to enable sending custom messages with SeQUeNCe's message package
class MsgType(Enum):
    ZERO = 0
    ONE = 1
    READY = 2
    
class Player(Enum):
    ALICE = 0
    BOB = 1

# Define a function to get the node a name refers to (for convenience)
def getNodeFromName(name: str, nodes: list[Node]):
    for node in nodes:
        if node.name == name:
            return node
    
    print("Node not found. Returning NoneType")
    return None

# Define custom protocols that define all three participants' behavior when the game starts
class RefereeProtocol(Protocol):
    def __init__(self, own: Node, name: str, tl: Timeline, players: list[Node], eff: int = 1, debug: bool = False, ge: bool = False,
                results=None):
        super().__init__(own, name)
        self.own = own
        self.own.protocols.append(self)
        self.tl = tl
        self.players = players
        self.inputs = []
        self.msgs_rec = 0
        self.keys = []
        self.result = None
        self.eff = eff
        self.debug = debug
        self.guar_ent = ge
        self.results=results

    def init(self):
        pass
    
    # Define a function to generate a random input (0 or 1) and send it to a player
    def sendBit(self, player_prot: str, player_node: str):
        bit = random.randint(0, 1)
        self.inputs.append(bit)
        msg = Message(MsgType(bit), player_prot)
        self.own.send_message(player_node, msg)
    
    # Define a function to handle player's responses
    def received_message(self, src: str, msg: Message):      
        self.msgs_rec += 1
        
        src_node = getNodeFromName(src, self.players)
        key = src_node.get_components_by_type('Memory')[0].qstate_key
        self.keys.append(key)
        circ = src_node.shared_circ
        
        #If both players have responded, the referee runs their circuit and measures their qubits
        if (self.msgs_rec == 2):   
            if (self.guar_ent):
                assert src_node.get_components_by_type('Memory')[0].entangled_memory['node_id'] != None, 'Entanglement generation failed'
            
            circ.measure(0)
            circ.measure(1)
            res = self.tl.quantum_manager.run_circuit(circ, self.keys, random.random())
            self.adjudicate_round(self.inputs, res)
            
    # Define a function to check if player responses met their win condition, return a win if so
    # def adjudicate_round(self, inp: list[int], res: dict[int, int]):       
    #     # Simulate noise in detector measurement
    #     if (random.random() > self.eff):           
    #         if self.debug:
    #             print('Alice\'s qubit, which was', res[0], ', was measured as 0.')    
    #         res[0] = 0
            
    #     if (random.random() > self.eff):
    #         if self.debug:
    #             print('Bob\'s qubit, which was', res[1], ', was measured as 0.')  
    #         res[1] = 0
        
    #     a_inp = inp[0]
    #     b_inp = inp[1]
    #     # a_res = res[0]
    #     # b_res = res[1]
    #     a_res = 1 - res[0]  # flip Alice’s bit
    #     b_res = 1 - res[1]  # flip Bob’s bit
    #     # print(f"Inp={inp}, Raw res={res}, XOR={a_res ^ b_res}, Target={a_inp & b_inp}")
    #     if ((a_res ^ b_res)== a_inp * b_inp):
    #         if self.debug:
    #             print('WIN! Inputs:', inp, 'Outputs:', res)
    #         self.result = True
    #     else:
    #         if self.debug:
    #             print('LOSS. Inputs:', inp, 'Outputs:', res)
    #         self.result = False


    #     key = (a_inp, b_inp)
    #     val = f"{a_res}{b_res}"
    #     self.results[key][val] += 1
    def adjudicate_round(self, inp: list[int], res: dict[int, int]):
    # channel/detector flip noise on RAW bits
        if random.random() > self.eff:
            res[0] ^= 1
        if random.random() > self.eff:
            res[1] ^= 1

        a_inp, b_inp = inp

        # configurable mapping flips (±1↔{0,1} convention)
        # FLIP_ALICE = True
        # FLIP_BOB = True
        # a_res = (1 - res[0]) if FLIP_ALICE else res[0]
        # b_res = (1 - res[1]) if FLIP_BOB   else res[1]

        a_res = res[0]
        b_res = res[1]

        target = a_inp & b_inp
        win = (a_res ^ b_res) == target
        self.result = win

        if self.debug:
            print(('WIN!' if win else 'LOSS.') +
                  f" Inputs: {inp} Outputs(bits): {{0: {a_res}, 1: {b_res}}} XOR={a_res ^ b_res} Target={target}")

        key = (a_inp, b_inp)
        val = f"{a_res}{b_res}"
        self.results[key][val] += 1


    # Define a getter function for other classes to get the result
    def get_result(self):
        assert self.result is not None, 'Result was NoneType. Has the game finished?'
        return self.result

    def compute_chsh_s(self):
        def expectation(counts):
            total = sum(counts.values())
            e = 0
            for outcome, count in counts.items():
            # Handle tuple (0,1) or string "01"
                if isinstance(outcome, str):
                    a, b = int(outcome[0]), int(outcome[1])
                else:
                    a, b = outcome
                if a == b:
                    e += count
                else:
                    e -= count
            if total == 0:
                return 0
            val = e / total
            if abs(val) > 1.0:
                print(f"[WARNING] Clipping E = {val} to [-1, 1]")
            return max(min(val, 1.0), -1.0)

        E = {k: expectation(v) for k, v in self.results.items()}
        S = abs(E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)])

    # --- Added: per-setting and global winrate logging ---
        # total_rounds = 0
        # total_wins = 0
        # print("\n[CHSH Stats]")
        # for k, counts in self.results.items():
        #     total = sum(counts.values())
        #     wins = sum(
        #     count for outcome, count in counts.items()
        #     if (int(outcome[0]) ^ int(outcome[1])) == (k[0] & k[1])
        #     )
        #     total_rounds += total
        #     total_wins += wins
        #     winrate_xy = wins / total if total else 0
        #     print(f"  Setting {k}: E = {E[k]:.3f}, winrate = {winrate_xy:.3f}, rounds = {total}")

        # global_winrate = total_wins / total_rounds if total_rounds else 0
        # print(f"Global S = {S:.4f}, Global winrate = {global_winrate:.4f}\n")
    # -----------------------------------------------------

        return S


    
class PlayerProtocol(Protocol):
    def __init__(self, own: Node, name: str, rec_name: str, rec_node: str):
        super().__init__(own, name)
        self.rec_name = rec_name
        self.rec_node = rec_node
        self.own = own
        own.protocols.append(self)
    
    def init(self):
        pass
    
    # Define a method to get an input from the referee, add the appropriate gates to the circuit,
    # Then signal to the referee that the player is ready
    def received_message(self, src: str, msg: Message):
        self.own.protocols[1].applyGate(msg)
        return_msg = Message(MsgType(2), self.rec_name)
        self.own.send_message(self.rec_node, return_msg)
        
# Updated AliceProtocol and BobProtocol for Circuit B alignment logic

class AliceProtocol(Protocol):
    def __init__(self, own: Node, name: str, theta1, theta2, delta=None, alpha=None):
        super().__init__(own, name)
        self.own = own
        self.theta1 = theta1
        self.theta2 = theta2
        self.delta = delta  # Impairment
        self.alpha = alpha  # Alignment angles
        own.protocols.append(self)

    def init(self):
        pass

    def applyGate(self, in_bit: Message):
        if self.delta is not None:
            self.own.shared_circ.gates.append(['rx', [self.own.reg], 2 * self.delta[0]])
            self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.delta[1]])
            self.own.shared_circ.gates.append(['rz', [self.own.reg], 2 * self.delta[2]])

        # Apply alignment gates (Circuit B)
        if self.alpha is not None:
            self.own.shared_circ.gates.append(['u1', [self.own.reg], self.alpha[0]])
            self.own.shared_circ.gates.append(['u2', [self.own.reg], self.alpha[1]])
            self.own.shared_circ.gates.append(['u3', [self.own.reg], self.alpha[2]])
        # Apply CHSH rotation depending on input
        if in_bit != None:
            if in_bit.msg_type == MsgType.ONE:
                self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.theta1])
            elif in_bit.msg_type == MsgType.ZERO:
                self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.theta2])
            else:
                pass

        # Apply impairment (delta)
        

    def received_message(self, src: str, msg: Message):
        pass


class BobProtocol(Protocol):
    def __init__(self, own: Node, name: str, theta1, theta2, delta=None, alpha=None):
        super().__init__(own, name)
        self.own = own
        self.theta1 = theta1
        self.theta2 = theta2
        self.delta = delta  # Impairment
        self.alpha = alpha  # Alignment angles
        own.protocols.append(self)

    def init(self):
        pass

    def applyGate(self, in_bit: Message):
        # Apply impairment (delta)
        if self.delta is not None:
            self.own.shared_circ.gates.append(['rx', [self.own.reg], 2 * self.delta[0]])
            self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.delta[1]])
            self.own.shared_circ.gates.append(['rz', [self.own.reg], 2 * self.delta[2]])

        # Apply alignment gates (Circuit B)
        if self.alpha is not None:
            self.own.shared_circ.gates.append(['u1', [self.own.reg], self.alpha[0]])
            self.own.shared_circ.gates.append(['u2', [self.own.reg], self.alpha[1]])
            self.own.shared_circ.gates.append(['u3', [self.own.reg], self.alpha[2]])
        # Apply CHSH rotation depending on input
        if in_bit != None:
            if in_bit.msg_type == MsgType.ONE:
                self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.theta1])
            elif in_bit.msg_type == MsgType.ZERO:
                self.own.shared_circ.gates.append(['ry', [self.own.reg], 2 * self.theta2])
            else:
                pass

        

    def received_message(self, src: str, msg: Message):
        pass
# This code creates a Game class to manage the creation and simulation of individual games

class Game:
    def __init__(self,debug: bool = False, fid: int = 1, freq: int = 1, eff: int = 1, 
                 wave: int = 1,guarantee_entanglement: bool = True,deltas = None):
        self.games = 100
        self.wins = 0
        self.debug = debug
        self.fidelity = fid
        self.efficiency = eff
        self.guar_ent = guarantee_entanglement
        self.frequency = freq
        self.wave = wave
        self.deltas = deltas
        self.referee = None
        self.results = {(x, y): {"00": 0, "01": 0, "10": 0, "11": 0} for x in [0, 1] for y in [0, 1]}
        
        if (not self.guar_ent):
            self.successful_entanglements = 0
        
    def setup(self):
        # Create the timeline for the simulation
        self.tl = Timeline()

        # Create a circuit for Alice and Bob to share
        self.shared_circ = CustomCircuit(size=2)

        # Create nodes for Alice, Bob, and a Bell State Management Node to generate entanglement
        self.a = PlayerNode('a', self.tl, self.shared_circ, 0, fid = self.fidelity)
        self.b = PlayerNode('b', self.tl, self.shared_circ, 1, fid = self.fidelity)

        # The referee's channel uses SeQUeNCe's built-in BSMNode
        self.ent_node = BSMNode('ent_node', self.tl, ['a', 'b'])

        # Set the efficiency of the BSM to 1, which means no errors
        self.bsm = self.ent_node.get_components_by_type('SingleAtomBSM')[0]
        self.bsm.update_detectors_params('efficiency', 1)

        # Create a node for the referee to get and receive bits
        self.r = Node('ref', self.tl)
        self.nodes = [self.a, self.b, self.ent_node, self.r]

        # Create quantum channels between Alice and Bob and the ref
        self.qcA = QuantumChannel('qcA', self.tl, attenuation = 0, distance = 10e1)
        self.qcB = QuantumChannel('qcB', self.tl, attenuation = 0, distance = 10e1)
        self.qcA.set_ends(self.a, self.ent_node.name)
        self.qcB.set_ends(self.b, self.ent_node.name)

        # Create classical channels between all existing nodes
        # Classical channels are one way only, so we have to make two channels for each connection
        for i in range (len(self.nodes)):
            for j in range(len(self.nodes)):
                if (i != j):
                    cc = ClassicalChannel('cc_%s_%s'%(self.nodes[i].name, self.nodes[j].name), self.tl, 1000, 1e8)
                    cc.set_ends(self.nodes[i], self.nodes[j].name)
    
    # Define a function to simulate the games and print the number won
    def start(self, games: int = 100,theta1=0,theta2=0,theta3=0,theta4=0,alphas_alice=None, alphas_bob=None):
        self.games = games
        
        for i in range(self.games):
            # Create nodes for every player and define entanglement protocol
            self.setup()
            
            # Add and run the entanglement protocol to each player node
            generate_entanglement(self.a, self.b, self.ent_node)
            self.tl.init()
            self.a.protocols[0].start()
            self.b.protocols[0].start()
            self.tl.run()

            mem_a = self.a.get_components_by_type('Memory')[0]
            
            # Keep rerunning the entanglement protocol until entanglement is successfully created
            if (self.guar_ent):
                while (mem_a.entangled_memory['node_id'] == None): 
                    generate_entanglement(self.a, self.b, self.ent_node)
                    self.a.protocols[0].start()
                    self.b.protocols[0].start()
                    self.tl.run()
            else:
                if (self.a.get_components_by_type('Memory')[0].entangled_memory['node_id'] != None):
                    self.successful_entanglements += 1
            
            # Play a round and store the result in a variable
            rnd = self.play_round(theta1,theta2,theta3,theta4,alphas_alice, alphas_bob)
            
            # If the result is a win, add one to the win counter
            if rnd:
                self.wins += 1
        
        if (not self.guar_ent):
            print('Successfully entangled Alice and Bob\'s qubits', self.successful_entanglements, 
                  'out of', self.games, 'games.')
            
        # print('Won', self.wins, 'out of', self.games, 'games.')
        # print(self.games)
        # print(self.wins)
        return (self.wins/self.games)
    # Define a function to send input bits to each player and return whether their outputs won the win condition
    def play_round(self,theta1,theta2,theta3,theta4,alphas_alice=None, alphas_bob=None):
        # Reset all participants' protocols
        self.preset()
        
        # Add protocols for each node to assign their behavior
        self.a.resource_manager.add_player_protocol('prot_r', 'ref')
        # # self.a.resource_manager.add_alice_protocol()
        self.b.resource_manager.add_player_protocol('prot_r', 'ref')
        # # self.b.resource_manager.add_bob_protocol()

        deltA = 1
        deltB = 1
        if self.deltas != None:
            deltA = self.deltas[0] 
            deltB = self.deltas[1]
        # print(deltA,deltB)
        self.a.resource_manager.add_alice_protocol(theta1,theta2,delta = deltA,alphas =alphas_alice)
        self.b.resource_manager.add_bob_protocol(theta3,theta4,delta = deltB,alphas =alphas_bob)
        
        # Add protocol to the referee to assign their behavior
        self.referee = RefereeProtocol(self.r, 'prot_r', self.tl, [self.a, self.b], eff = self.efficiency, 
                                 debug = self.debug, ge = self.guar_ent,results=self.results)

        # Send bits to both Alice and Bob, who will respond automatically
        self.r.protocols[0].sendBit('a.pp', 'a')
        self.r.protocols[0].sendBit('b.pp', 'b')
        self.tl.run()
        
        #Get whether Alice and Bob won or lost, and return it to the start function
        return self.referee.get_result()

    def calibration(self,theta1,theta2,theta3,theta4,alpha_alice=None,alpha_bob=None):
        self.setup()
        
        self.preset()
        
        self.a.resource_manager.add_player_protocol('prot_r', 'ref') 
        self.b.resource_manager.add_player_protocol('prot_r', 'ref')

        deltA = 1
        deltB = 1
        if self.deltas != None:
            deltA = self.deltas[0] 
            deltB = self.deltas[1]
            
        self.a.resource_manager.add_alice_protocol(theta1,theta2,delta = deltA,alphas = alpha_alice)
        self.b.resource_manager.add_bob_protocol(theta3,theta4,delta = deltB,alphas = alpha_bob)
        
        # Add protocol to the referee to assign their behavior
        self.a.protocols[1].applyGate(None)
        self.b.protocols[1].applyGate(None)
        
        #Get whether Alice and Bob won or lost, and return it to the start function
        self.shared_circ.measure(0)  # Alice
        self.shared_circ.measure(1)  # Bob

        # Get qstate keys from memories
        key_a = self.a.get_components_by_type('Memory')[0].qstate_key
        key_b = self.b.get_components_by_type('Memory')[0].qstate_key
        keys = [key_a,key_b ]

        # Run the circuit and get result
        res = self.tl.quantum_manager.run_circuit(self.shared_circ, keys, random.random())
        
        
        return res 
    def e91_round(self,
                  a_basis_idx: int,
                  b_basis_idx: int,
                  alphas_alice=None,
                  alphas_bob=None):
        """
        Run ONE E91 round:
          1. Create entanglement between Alice and Bob.
          2. Apply delta + alignment (Circuit B) on each side.
          3. Apply E91 measurement basis rotations.
          4. Measure Z on both qubits.
        Returns:
          (a_basis_idx, b_basis_idx, a_bit, b_bit)
        """

        # --- 1. Fresh setup + entanglement (same pattern as start()) ---
        self.setup()

        # Entanglement-generation protocols
        generate_entanglement(self.a, self.b, self.ent_node)
        self.tl.init()
        self.a.protocols[0].start()
        self.b.protocols[0].start()
        self.tl.run()

        mem_a = self.a.get_components_by_type('Memory')[0]

        if self.guar_ent:
            while mem_a.entangled_memory['node_id'] is None:
                generate_entanglement(self.a, self.b, self.ent_node)
                self.a.protocols[0].start()
                self.b.protocols[0].start()
                self.tl.run()

        # --- 2. Clear protocols & gates, keep entangled state (same idea as preset()) ---
        self.preset()  # clears self.a.protocols, self.b.protocols, self.r.protocols
                       # and self.shared_circ.gates, but NOT the memories

        # Re-attach a PlayerProtocol (not strictly needed for E91 logic,
        # but keeps structure consistent with your existing design)
        self.a.resource_manager.add_player_protocol('prot_r', 'ref')
        self.b.resource_manager.add_player_protocol('prot_r', 'ref')

        # --- 3. Add Alice/Bob misalignment + alignment protocols (no CHSH bit) ---
        deltA = [0,0,0]
        deltB = [0,0,0]
        if self.deltas is not None:
            deltA = self.deltas[0]
            deltB = self.deltas[1]

        # Pass theta1=theta2=0 because CHSH-specific RY(±θ) is not used in E91
        self.a.resource_manager.add_alice_protocol(theta1=0,
                                                   theta2=0,
                                                   delta=deltA,
                                                   alphas=alphas_alice)
        self.b.resource_manager.add_bob_protocol(theta1=0,
                                                 theta2=0,
                                                 delta=deltB,
                                                 alphas=alphas_bob)

        # Apply only delta + alignment (we pass None so no CHSH rotation happens)
        self.a.protocols[1].applyGate(None)
        self.b.protocols[1].applyGate(None)

        # --- 4. Apply the E91 measurement basis rotations ---
        phi_a = E91_ALICE_BASES[a_basis_idx]
        phi_b = E91_BOB_BASES[b_basis_idx]

        # Rotate Alice’s and Bob’s measurement bases (qubits 0 and 1 in shared_circ)
        self.shared_circ.gates.append(['ry', [0], 2 * phi_a])
        self.shared_circ.gates.append(['ry', [1], 2 * phi_b])

        # --- 5. Measure both qubits in Z ---
        self.shared_circ.measure(0)  # Alice
        self.shared_circ.measure(1)  # Bob

        # Use the same key-extraction pattern as calibration()
        key_a = self.a.get_components_by_type('Memory')[0].qstate_key
        key_b = self.b.get_components_by_type('Memory')[0].qstate_key
        keys = [key_a, key_b]

        res = self.tl.quantum_manager.run_circuit(self.shared_circ,
                                                  keys,
                                                  random.random())

        a_bit = int(res[0])
        b_bit = int(res[1])

        return a_basis_idx, b_basis_idx, a_bit, b_bit

    def run_e91(self,
                num_rounds: int = 10_000,
                alphas_alice=None,
                alphas_bob=None,
                verbose: bool = True):
        """
        Run a full E91 protocol:
          - Randomly choose bases for Alice and Bob (3 each).
          - For each round, do entanglement + misalignment + E91 basis measurement.
          - Sift:
              * (A0, B0) → raw key bits
              * (A1,B0), (A1,B1), (A2,B0), (A2,B1) → CHSH samples
          - Compute CHSH S-value from those samples.

        Returns:
          key_alice: np.array of Alice's sifted key bits
          key_bob:   np.array of Bob's sifted key bits
          S_value:   CHSH S
          chsh_data: dict[(x,y)] -> list[(a_bit, b_bit)]
        """

        all_a_basis = []
        all_b_basis = []
        all_a_bits = []
        all_b_bits = []

        # --- 1. Collect raw data over many rounds ---
        for _ in range(num_rounds):
            a_basis = np.random.randint(0, len(E91_ALICE_BASES))  # 0,1,2
            b_basis = np.random.randint(0, len(E91_BOB_BASES))    # 0,1,2

            a_b, b_b, a_bit, b_bit = self.e91_round(
                a_basis_idx=a_basis,
                b_basis_idx=b_basis,
                alphas_alice=alphas_alice,
                alphas_bob=alphas_bob
            )

            all_a_basis.append(a_b)
            all_b_basis.append(b_b)
            all_a_bits.append(a_bit)
            all_b_bits.append(b_bit)

        all_a_basis = np.array(all_a_basis)
        all_b_basis = np.array(all_b_basis)
        all_a_bits  = np.array(all_a_bits)
        all_b_bits  = np.array(all_b_bits)

        # --- 2. Sift for key: E91_KEY_PAIR (A0,B0) ---
        key_mask = (all_a_basis == E91_KEY_PAIR[0]) & (all_b_basis == E91_KEY_PAIR[1])
        key_alice = all_a_bits[key_mask]
        key_bob   = all_b_bits[key_mask]

        # --- 3. Sift for CHSH samples using E91_CHSH_MAP ---
        chsh_data = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

        for a_b, b_b, a_bit, b_bit in zip(all_a_basis, all_b_basis, all_a_bits, all_b_bits):
            key_pair = (a_b, b_b)
            if key_pair in E91_CHSH_MAP:
                x, y = E91_CHSH_MAP[key_pair]
                chsh_data[(x, y)].append((a_bit, b_bit))

        # --- 4. Compute E(x,y) and CHSH S ---
        def expectation(pairs):
            if len(pairs) == 0:
                return 0.0
            e = 0
            for a_bit, b_bit in pairs:
                if a_bit == b_bit:
                    e += 1
                else:
                    e -= 1
            val = e / len(pairs)
            # safety clip
            return max(min(val, 1.0), -1.0)

        E = {k: expectation(v) for k, v in chsh_data.items()}
        S = abs(E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)])

        keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
        Es   = [E[k] for k in keys]

        best_S = 0.0
        best_signs = None

        for signs in itertools.product([-1, 1], repeat=4):
        # Only patterns with an odd number of '-' (product = -1) are valid CHSH forms
            prod = signs[0] * signs[1] * signs[2] * signs[3]
            if prod != -1:
                continue
            S_val = abs(sum(s * e for s, e in zip(signs, Es)))
            # print(S_val,best_S)
            if S_val > best_S:
                best_S = S_val
                best_signs = signs

        S_opt = best_S
        if verbose:
            print("\n[E91 Results]")
            print(f"  Total rounds            : {num_rounds}")
            print(f"  Key rounds (A0,B0)      : {key_alice.size}")
            print("  CHSH stats:")
            for k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                print(f"    setting {k}: samples = {len(chsh_data[k])}, E = {E[k]:.3f}")
            print(f"  CHSH S-value            : {S_opt:.4f}")
            if S_opt > 2.0:
                print("  -> Bell violation (secure E91 key in principle).")
            else:
                print("  -> No Bell violation; key should be discarded.")

        return key_alice, key_bob, S_opt, chsh_data

    # Define a function to clear player protocols and circuits to ensure the game runs from scratch
    def preset(self):
        self.a.protocols = []
        self.b.protocols = []
        self.r.protocols = []
        self.shared_circ.gates = []
        self.shared_circ.measured_qubits = []