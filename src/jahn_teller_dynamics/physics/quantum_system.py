"""
Quantum system tree module for hierarchical quantum system representation.

This module provides classes for building and managing hierarchical quantum systems
represented as trees. It supports:
- Tree-based node structures for organizing quantum subsystems
- Quantum system nodes with operators and basis states
- Tensor product operations for composite systems
- Operator creation and management across subsystems

Main classes:
- node: Basic tree node structure
- tree: Tree container with root node
- quantum_system_node: Quantum system with operators and basis states
- quantum_system_tree: Tree of quantum systems with operator management

This module is fundamental to the Jahn-Teller dynamics calculations, providing
the structure for organizing electron, spin, and phonon subsystems.
"""

import itertools
import numpy as np
import math
import jahn_teller_dynamics.math.maths as maths
import copy
import jahn_teller_dynamics.math.matrix_mechanics as mm
import operator
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING

from collections import namedtuple

if TYPE_CHECKING:
    from typing import Union





class node:
    """
    Basic tree node class for hierarchical structures.
    
    Represents a node in a tree structure with an ID and optional children.
    Provides methods for tree traversal, node finding, and depth calculation.
    """
    
    def __init__(self, id: str, children: Optional[List['node']] = None) -> None:
        """
        Initialize a tree node.
        
        Args:
            id: Unique identifier for the node
            children: Optional list of child nodes
        """
        self.id = id
        if children is None:
            self.children: List['node'] = []
        else:
            self.children = children
    
    def has_child(self) -> bool:
        """Check if node has children."""
        return len(self.children) > 0

    def add_child(self, child: 'node') -> None:
        """Add a child node."""
        self.children.append(child)

    def get_depth(self, target_node: Any) -> int:
        """
        Get depth of target_node in the tree.
        
        Args:
            target_node: Node to find depth of
            
        Returns:
            int: Depth of the node, or -1 if not found
        """
        depth = 0
        found = self._get_depth_impl(target_node, depth)
        return found if found >= 0 else -1
    
    def _get_depth_impl(self, target_node: Any, current_depth: int) -> int:
        """Helper method for get_depth."""
        if self == target_node or self.id == target_node:
            return current_depth
        
        if self.has_child():
            for child in self.children:
                result = child._get_depth_impl(target_node, current_depth + 1)
                if result >= 0:
                    return result
        
        return -1

    def find_leaves_avoid(self, id: str) -> Tuple[List['node'], Optional['node'], List['node']]:
        """
        Find all leaves while avoiding a specific node.
        
        Args:
            id: ID of the node to avoid (but still include in result)
            
        Returns:
            Tuple containing:
            - left_side_leaves: List of leaves before the avoided node
            - avoided_node: The node with the given id (or None if not found)
            - right_side_leaves: List of leaves after the avoided node
        """
        res = []
        self.find_leaves_avoid_imp(id, res)

        avoid_index = 0
        avoided_node = None

        for i in range(len(res)):
            if res[i].id == id:
                avoid_index = i
                avoided_node = res[i]
                break

        left_side_leaves = res[0:avoid_index]
        right_side_leaves = res[avoid_index+1:]

        return left_side_leaves, avoided_node, right_side_leaves

    def find_leaves_avoid_imp(self, id: str, res: List['node']) -> None:
        """
        Helper method for find_leaves_avoid - recursively collects leaves.
        
        Args:
            id: ID of node to mark (but still include)
            res: List to append nodes to
        """
        if self.id == id:
            res.append(self)
        elif self.has_child():
            for child in self.children:
                child.find_leaves_avoid_imp(id, res)
        else:
            res.append(self)

    def find_leaves(self) -> List['node']:
        """Find all leaf nodes in the tree."""
        res = []
        self.find_leaves_imp(res)
        return res

    def find_leaves_imp(self, res: List['node']) -> None:
        """
        Helper method for find_leaves - recursively collects all leaf nodes.
        
        Args:
            res: List to append leaf nodes to
        """
        if self.has_child():
            for child in self.children:
                child.find_leaves_imp(res)
        else:
            res.append(self)

    def find_node(self, id: str) -> Optional['node']:
        """
        Find node with given id in the tree.
        
        Args:
            id: Node id to search for
            
        Returns:
            Optional[node]: First matching node, or None if not found.
            If multiple nodes have the same id, returns the first one found.
            
        Note:
            This method returns None if the node is not found, following a
            "graceful failure" pattern. For methods that raise exceptions
            on failure, see find_operator() and create_operator().
        """
        res = []
        depths = []

        self.find_node_imp(id, res, depths, new_depth_0=0)
        if len(res) == 0:
            return None
        elif len(res) == 1:
            return res[0]
        else:
            # Multiple nodes found - return the first one (closest to root)
            # Could also raise a warning here
            return res[0]

    def find_node_imp(self, data: str, res: List['node'], depths: List[int], new_depth_0: int) -> None:
        """
        Helper method for find_node - recursively searches tree.
        
        Args:
            data: Node ID to search for
            res: List to append matching nodes to
            depths: List to append node depths to
            new_depth_0: Current depth in tree
        """
        new_depth = copy.deepcopy(new_depth_0)
        
        if self.id == data:
            res.append(self)
            depths.append(new_depth)
            
        if self.has_child():
            new_depth += 1
            for child in self.children:
                child.find_node_imp(data, res, depths, new_depth)

    def get_nodes(self, depth: int) -> List['node']:
        """
        Get all nodes at a specific depth in the tree.
        
        Args:
            depth: Target depth (0 = root)
            
        Returns:
            List of nodes at the specified depth
        """
        res = []
        self.get_nodes_imp(depth, 0, res)
        return res

    def get_nodes_imp(self, depth: int, curr_depth: int, res: List['node']) -> None:
        """
        Helper method for get_nodes - recursively collects nodes at target depth.
        
        Args:
            depth: Target depth to find nodes at
            curr_depth: Current depth in recursion
            res: List to append nodes to
        """
        if depth == curr_depth:
            res.append(self)
        else:
            if self.has_child():
                curr_depth += 1
                for child in self.children:
                    child.get_nodes_imp(depth, copy.deepcopy(curr_depth), res)

    def __repr__(self) -> str:
        """Return string representation of the node (its ID)."""
        return self.id


class tree:
    """
    Tree container class for managing hierarchical node structures.
    
    Wraps a root node and provides methods for tree manipulation.
    """
    
    def __init__(self, root_node: node) -> None:
        """
        Initialize tree with a root node.
        
        Args:
            root_node: The root node of the tree
        """
        self.root_node = root_node

    def insert_node(self, parent_node_id: str, new_child: node) -> None:
        """
        Insert a new child node into the tree.
        
        Args:
            parent_node_id: ID of the parent node
            new_child: New child node to insert
            
        Note:
            If a node with the same id already exists in the tree, 
            insertion is silently skipped (preserves original behavior).
            
        Raises:
            ValueError: If parent node not found
        """
        # Only insert if node doesn't already exist (preserve original behavior)
        if self.root_node.find_node(new_child.id) is None:
            parent_node = self.root_node.find_node(parent_node_id)
            if parent_node is None:
                raise ValueError(f"Parent node '{parent_node_id}' not found")
            parent_node.add_child(new_child)


class quantum_system_node(node):
    """
    Quantum system node representing a quantum subsystem.
    
    Extends the basic node class with quantum mechanical properties:
    - Basis states (Hilbert space basis)
    - Operators (quantum operators acting on the system)
    - Dimension (Hilbert space dimension)
    
    Can be organized in a tree structure to represent composite quantum systems.
    """

    @staticmethod
    def create_2D_orbital_system_node() -> 'quantum_system_node':
        """
        Create a 2D orbital system node with E⊗e Jahn-Teller basis.
        
        Returns:
            quantum_system_node: Orbital system with Pauli operators
        """
        el_sys_ops = {}
        # Use SQRT_2 constant from matrix_mechanics module
        sqrt_2_inv = 1.0 / mm.SQRT_2
        
        # Create basis vectors for E⊗e system
        # Note: (-2)**0.5 = i*sqrt(2), so we use complex(0, sqrt_2)
        b1 = mm.ket_vector([-sqrt_2_inv, -complex(0.0, 1.0) * sqrt_2_inv])
        b2 = mm.ket_vector([sqrt_2_inv, -complex(0.0, 1.0) * sqrt_2_inv])


        bs = [b1, b2]
        

        el_sys_ops['X_orb'] = mm.MatrixOperator.pauli_x_mx_op()
        el_sys_ops['Y_orb'] = mm.MatrixOperator.pauli_y_mx_op()
        el_sys_ops['Z_orb'] = mm.MatrixOperator.pauli_z_mx_op()

        el_sys_ops['X_Alt'] = mm.MatrixOperator(maths.Matrix(np.matrix([[1.0+0.0j, 0.0j], [0.0j, 0.0j]],dtype=np.complex128)))
        el_sys_ops['Y_Alt'] = mm.MatrixOperator(maths.Matrix(np.matrix([[0.0j, 0.0j], [0.0j, 1.0+0.0j]],dtype=np.complex128)))

        el_sys_ops['Lz'] = mm.MatrixOperator.pauli_z_mx_op().to_new_basis(bs)
        el_sys_ops['Lx'] = mm.MatrixOperator.pauli_x_mx_op().to_new_basis(bs)
        el_sys_ops['Ly'] = mm.MatrixOperator.pauli_y_mx_op().to_new_basis(bs)

        to_cmp_basis_trf = mm.MatrixOperator.basis_trf_matrix(bs)

        el_sys_ops['C_tr'] = to_cmp_basis_trf


        orbital_system = quantum_system_node('orbital_system', base_states=mm.hilber_space_bases().from_qm_nums_list([ ['ex'],[ 'ey']],
                                                                        qm_nums_names=['orbital'])  ,operators=el_sys_ops, dim= 2)


        return orbital_system
    @staticmethod
    def create_2D_complex_orbital_system_node() -> 'quantum_system_node':
        """
        Create a 2D complex orbital system node with E⊗e Jahn-Teller basis.
        
        Returns:
            quantum_system_node: Complex orbital system with Pauli operators
        """
        el_sys_ops = {}
        # Note: sqrt_2_inv not needed for this method, but kept for consistency if needed later

        orbital_system = quantum_system_node('orbital_system', base_states=mm.hilber_space_bases().from_qm_nums_list([ ['e+'],[ 'e-']],
                                                                        qm_nums_names=['orbital'])  ,operators=el_sys_ops, dim= 2)

        return orbital_system

    @staticmethod
    def create_spin_system_node() -> 'quantum_system_node':
        """
        Create a spin-1/2 system node.
        
        Returns:
            quantum_system_node: Spin system with S_x, S_y, S_z operators
        """
    
        spin_sys_ops = {}

        spin_sys_ops['Sz'] = 0.5*mm.MatrixOperator.pauli_z_mx_op()
        spin_sys_ops['Sy'] = 0.5*mm.MatrixOperator.pauli_y_mx_op()
        spin_sys_ops['Sx'] = 0.5*mm.MatrixOperator.pauli_x_mx_op()


        spin_sys = quantum_system_node('spin_system', mm.hilber_space_bases().from_qm_nums_list([['up'], ['down']] , qm_nums_names=['spin']) , operators=spin_sys_ops)

        return spin_sys

    def create_id_op(self, matrix_type: type = maths.Matrix) -> mm.MatrixOperator:
        """Create identity operator for this system."""
        id_op = mm.MatrixOperator.create_id_matrix_op(self.dim, matrix_type=matrix_type)
        return id_op

    def __init__(self, id: str, base_states: Optional[mm.hilber_space_bases] = None, operators: Optional[Dict[str, mm.MatrixOperator]] = None, children: Optional[List['quantum_system_node']] = None, dim: int = 1) -> None:
        """
        Initialize a quantum system node.
        
        Args:
            id: Unique identifier for the system
            base_states: Hilbert space basis for this system (optional)
            operators: Dictionary of quantum operators (optional)
            children: List of child quantum system nodes (optional)
            dim: Dimension of the Hilbert space (used if base_states is None)
            
        Note:
            If children are provided, the Hilbert space is automatically created
            from the tensor product of children's basis states.
        """
        node.__init__(self, id, children)
        self.operators: Dict[str, mm.MatrixOperator] = operators if operators is not None else {}
        self.base_states: Optional[mm.hilber_space_bases] = base_states
        if self.base_states is not None:
            self.dim = self.base_states.dim
        else:
            self.dim = dim
        if len(self.children) >= 1:
            self.create_hilbert_space()

    def create_hilbert_space(self) -> None:
        """Create Hilbert space from children's basis states."""
        simple_systems = self.find_leaves()

        children_system_bases = [x.base_states for x in simple_systems if x.base_states is not None]

        self.base_states = mm.hilber_space_bases.kron_hilber_spaces(children_system_bases)
        self.dim = self.base_states.dim

    def find_operator(self, operator_id: str) -> str:
        """
        Find the system node containing the specified operator.
        
        Args:
            operator_id: Name of the operator to find
            
        Returns:
            str: ID of the system node containing the operator
            
        Raises:
            ValueError: If operator is not found in any subsystem
            
        Note:
            This method raises ValueError on failure, following an "explicit error"
            pattern. This is intentional - operator lookup failures are considered
            programming errors that should be caught immediately. For methods that
            return None on failure, see find_node().
        """
        res = []
        depths = []

        self.find_operator_imp(operator_id, res, depths, new_depth_0=0)
        if len(res) == 0:
            raise ValueError(f"Operator '{operator_id}' not found in any subsystem")
        return res[0].id

    def find_operator_imp(self, operator_id: str, res: List['quantum_system_node'], depths: List[int], new_depth_0: int) -> None:
        """
        Helper method for find_operator - recursively searches for operator.
        
        Args:
            operator_id: Name of operator to find
            res: List to append matching system nodes to
            depths: List to append node depths to
            new_depth_0: Current depth in tree
        """
        new_depth = copy.deepcopy(new_depth_0)
        
        if operator_id in self.operators:
            res.append(self)
            depths.append(new_depth)
            
        if self.has_child():
            new_depth += 1
            for child in self.children:
                child.find_operator_imp(operator_id, res, depths, new_depth)



    def create_operator(self, operator_id: str = '', operator_system_id: str = '') -> mm.MatrixOperator:
        """
        Create operator tensor product: I_left ⊗ op ⊗ I_right.
        
        Args:
            operator_id: Name of the operator
            operator_system_id: ID of the system containing the operator
            
        Returns:
            MatrixOperator: Tensor product operator
            
        Raises:
            ValueError: If operator or system not found
        """

        if operator_system_id == '':
            operator_system_id = self.find_operator(operator_id)

        left_systems, system, right_systems = self.find_leaves_avoid(operator_system_id)
        
        if system is None:
            raise ValueError(f"System '{operator_system_id}' not found")
        
        if operator_id not in system.operators:
            raise ValueError(f"Operator '{operator_id}' not found in system '{operator_system_id}'")

        op = system.operators[operator_id]


        left_dims = list(map(lambda x: x.dim, left_systems))
        left_dim = list(itertools.accumulate(left_dims, operator.mul))[-1] if left_dims != [] else 1

        right_dims = list(map(lambda x: x.dim, right_systems))
        right_dim = list(itertools.accumulate(right_dims, operator.mul))[-1] if right_dims != [] else 1





        I_left = mm.MatrixOperator.create_id_matrix_op(dim = left_dim)
        I_right = mm.MatrixOperator.create_id_matrix_op(dim = right_dim)

        return I_left**op**I_right


class quantum_system_tree(tree):
    """
    Quantum system tree for managing hierarchical quantum systems.
    
    Extends the basic tree class with quantum-specific operations:
    - Operator creation and management
    - Basis transformations
    - Automatic Hilbert space updates when nodes are inserted
    """

    def create_basis_trf_matrix(self, basis_name: str) -> Tuple[mm.MatrixOperator, mm.hilber_space_bases]:
        """
        Create basis transformation matrix for the entire system.
        
        Args:
            basis_name: Name of the basis to transform to
            
        Returns:
            Tuple containing:
            - Transformation matrix operator
            - New Hilbert space basis
        """
        leaf_systems: List[quantum_system_node] = self.root_node.find_leaves()

        basis_trf_matrixes = [leaf_system.base_states.create_trf_op(basis_name) for leaf_system in leaf_systems]

        new_hilbert_space = [leaf_system.base_states.base_vectors[basis_name][1] for leaf_system in leaf_systems]

        return list(itertools.accumulate(basis_trf_matrixes, lambda x, y: x**y))[-1], list(itertools.accumulate(new_hilbert_space, lambda x, y: x**y))[-1]

    def __init__(self, root_quantum_system_node: quantum_system_node) -> None:
        """
        Initialize quantum system tree with a root node.
        
        Args:
            root_quantum_system_node: Root quantum system node
        """
        self.root_node = root_quantum_system_node
    
    def insert_node(self, parent_node_id: str, new_child: quantum_system_node) -> None:
        """
        Insert a new child node and update Hilbert spaces.
        
        Args:
            parent_node_id: ID of the parent node
            new_child: New quantum system node to insert
            
        Note:
            After insertion, Hilbert spaces are automatically recalculated
            for both the root node and the parent node.
            
        Raises:
            ValueError: If parent node not found
        """
        super().insert_node(parent_node_id, new_child)
        
        self.root_node.create_hilbert_space()
        parent_node = self.find_subsystem(parent_node_id)
        if parent_node is not None:
            parent_node.create_hilbert_space()

    def create_operator(self, operator_id: str, subsys_id: str = '', operator_sys: str = '') -> mm.MatrixOperator:
        """
        Create operator tensor product at root or subsystem level.
        
        Args:
            operator_id: Name of the operator to create
            subsys_id: ID of subsystem to create operator in (empty = root level)
            operator_sys: ID of system containing the operator (auto-detected if empty)
            
        Returns:
            MatrixOperator: Tensor product operator I_left ⊗ op ⊗ I_right
            
        Raises:
            ValueError: If operator or system not found
        """
        if subsys_id == '':
            return self.root_node.create_operator(operator_id=operator_id, operator_system_id=operator_sys)
        else:
            subsys_node = self.root_node.find_node(subsys_id)
            if subsys_node is None:
                raise ValueError(f"Subsystem '{subsys_id}' not found")
            return subsys_node.create_operator(operator_id=operator_id, operator_system_id=operator_sys)

    def find_subsystem(self, subsystem_id: str) -> Optional['quantum_system_node']:
        """
        Find subsystem node with given id.
        
        Args:
            subsystem_id: ID of the subsystem to find
            
        Returns:
            Optional[quantum_system_node]: Subsystem node if found, None otherwise
        """
        return self.root_node.find_node(subsystem_id)