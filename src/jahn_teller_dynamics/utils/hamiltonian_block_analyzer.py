"""
Utility for analyzing Hamiltonian operator block structure.

This module provides functions to analyze the block-diagonal structure of
Hamiltonian operators and map blocks to corresponding basis states.
"""

from typing import List, Dict, Tuple, Optional, Union
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm


class BlockAnalysis:
    """Container for block analysis results."""
    
    def __init__(self):
        self.blocks: List[Dict] = []
        self.total_blocks: int = 0
        self.total_dimension: int = 0
        self.block_dimensions: List[int] = []
    
    def __repr__(self) -> str:
        return f"BlockAnalysis({self.total_blocks} blocks, dim={self.total_dimension})"


def analyze_hamiltonian_blocks(
    hamiltonian: mm.MatrixOperator,
    hilbert_space_bases: mm.hilber_space_bases,
    convert_to_sparse: bool = True
) -> BlockAnalysis:
    """
    Analyze block structure of a Hamiltonian operator.
    
    This function finds the block-diagonal structure of the Hamiltonian
    and maps each block to the corresponding basis states from the Hilbert space.
    
    Args:
        hamiltonian: MatrixOperator representing the Hamiltonian
        hilbert_space_bases: hilber_space_bases object containing the basis states
        convert_to_sparse: If True, convert dense matrices to sparse for analysis
        
    Returns:
        BlockAnalysis object containing block information
        
    Raises:
        ImportError: If networkx is not available (required for block analysis)
        ValueError: If matrix cannot be analyzed
    """
    # Check if networkx is available
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for block analysis. Install with: pip install networkx")
    
    # Get the matrix
    matrix = hamiltonian.matrix
    
    # Convert to sparse if needed
    if isinstance(matrix, maths.Matrix):
        if convert_to_sparse:
            sparse_matrix = matrix.to_sparse_matrix()
        else:
            raise ValueError("Dense matrices cannot be analyzed for blocks. Set convert_to_sparse=True or use a sparse matrix.")
    elif isinstance(matrix, maths.SparseMatrix):
        sparse_matrix = matrix
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")
    
    # Get blocks using the existing method
    try:
        sp_mx_blocks, new_basis_order = sparse_matrix.get_sparse_blocks_matrixes()
    except Exception as e:
        raise ValueError(f"Failed to get blocks: {e}")
    
    # Create analysis result
    analysis = BlockAnalysis()
    analysis.total_dimension = sparse_matrix.dim
    analysis.total_blocks = len(sp_mx_blocks)
    
    # Group basis indices by block
    # new_basis_order contains the reordered indices
    # We need to map these back to blocks
    block_indices: List[List[int]] = []
    current_idx = 0
    
    for block in sp_mx_blocks:
        block_dim = block.dim
        block_indices.append(list(range(current_idx, current_idx + block_dim)))
        analysis.block_dimensions.append(block_dim)
        current_idx += block_dim
    
    # Now map block indices to original basis indices and states
    for block_idx, (block, block_basis_indices) in enumerate(zip(sp_mx_blocks, block_indices)):
        # Map reordered indices to original indices
        original_indices = [new_basis_order[i] for i in block_basis_indices]
        
        # Get corresponding ket states
        ket_states = []
        for orig_idx in original_indices:
            if orig_idx < len(hilbert_space_bases._ket_states):
                ket_states.append(hilbert_space_bases._ket_states[orig_idx])
            else:
                ket_states.append(None)
        
        # Get quantum numbers for each state
        quantum_numbers = []
        state_strings = []
        for ket_state in ket_states:
            if ket_state is not None:
                qm_nums = ket_state.qm_nums if hasattr(ket_state, 'qm_nums') else None
                quantum_numbers.append(qm_nums)
                # Get string representation of state
                state_str = str(ket_state) if hasattr(ket_state, '__str__') else None
                state_strings.append(state_str)
            else:
                quantum_numbers.append(None)
                state_strings.append(None)
        
        # Store block information
        block_info = {
            'block_index': block_idx,
            'dimension': block_dim,
            'original_basis_indices': original_indices,
            'reordered_basis_indices': block_basis_indices,
            'ket_states': ket_states,
            'quantum_numbers': quantum_numbers,
            'state_strings': state_strings,
            'sparse_matrix': block
        }
        
        analysis.blocks.append(block_info)
    
    return analysis


def print_block_analysis(
    analysis: BlockAnalysis,
    hilbert_space_bases: mm.hilber_space_bases,
    max_states_per_block: int = 10,
    show_matrix_stats: bool = False
) -> None:
    """
    Print formatted block analysis results.
    
    Args:
        analysis: BlockAnalysis object from analyze_hamiltonian_blocks
        hilbert_space_bases: hilber_space_bases object (for state names)
        max_states_per_block: Maximum number of states to show per block
        show_matrix_stats: If True, show matrix statistics for each block
    """
    print("=" * 70)
    print("Hamiltonian Block Structure Analysis")
    print("=" * 70)
    print(f"\nTotal dimension: {analysis.total_dimension}")
    print(f"Number of blocks: {analysis.total_blocks}")
    print(f"Block dimensions: {analysis.block_dimensions}")
    print()
    
    # Get quantum number names if available
    qm_names = hilbert_space_bases.qm_nums_names if hasattr(hilbert_space_bases, 'qm_nums_names') else []
    
    for block_info in analysis.blocks:
        block_idx = block_info['block_index']
        block_dim = block_info['dimension']
        original_indices = block_info['original_basis_indices']
        ket_states = block_info['ket_states']
        quantum_numbers = block_info['quantum_numbers']
        state_strings = block_info.get('state_strings', [None] * len(ket_states))
        
        print(f"{'='*70}")
        print(f"Block {block_idx + 1} / {analysis.total_blocks}")
        print(f"{'='*70}")
        print(f"Dimension: {block_dim}")
        
        if show_matrix_stats and isinstance(block_info['sparse_matrix'], maths.SparseMatrix):
            sp_matrix = block_info['sparse_matrix']
            if hasattr(sp_matrix.matrix, 'nnz'):
                nnz = sp_matrix.matrix.nnz
                total = block_dim * block_dim
                sparsity = (1.0 - nnz / total) * 100 if total > 0 else 0
                print(f"Non-zero elements: {nnz:,} / {total:,} ({sparsity:.2f}% sparse)")
        
        # Analyze quantum numbers to find common patterns
        if quantum_numbers and any(qm is not None for qm in quantum_numbers):
            # Find unique quantum number values for each position
            unique_qm_values = {}
            for qm_nums in quantum_numbers:
                if qm_nums is not None:
                    for i, val in enumerate(qm_nums):
                        if i not in unique_qm_values:
                            unique_qm_values[i] = set()
                        unique_qm_values[i].add(val)
            
            # Print quantum number ranges
            if qm_names and unique_qm_values:
                print(f"\nQuantum number ranges in this block:")
                for i, name in enumerate(qm_names):
                    if i in unique_qm_values:
                        values = sorted(unique_qm_values[i])
                        if len(values) <= 10:
                            print(f"  {name}: {values}")
                        else:
                            print(f"  {name}: {values[0]} to {values[-1]} ({len(values)} unique values)")
        
        print(f"\nBasis states in this block ({min(len(original_indices), max_states_per_block)} of {len(original_indices)} shown):")
        print()
        
        # Show states
        for i, (orig_idx, ket_state, qm_nums, state_str) in enumerate(zip(
            original_indices[:max_states_per_block],
            ket_states[:max_states_per_block],
            quantum_numbers[:max_states_per_block],
            state_strings[:max_states_per_block] if state_strings else [None] * len(original_indices[:max_states_per_block])
        )):
            # Get state representation
            if state_str:
                display_str = state_str
            elif ket_state is not None:
                display_str = str(ket_state) if hasattr(ket_state, '__str__') else f"State {orig_idx}"
            else:
                display_str = f"State {orig_idx} (no ket_state)"
            
            print(f"  [{orig_idx:4d}] {display_str}")
            
            # Show quantum numbers if available and different from state string
            if qm_nums is not None and qm_names and state_str is None:
                qm_str = ", ".join([f"{name}={val}" for name, val in zip(qm_names, qm_nums)])
                print(f"        QN: {qm_str}")
        
        if len(original_indices) > max_states_per_block:
            print(f"  ... ({len(original_indices) - max_states_per_block} more states)")
        
        print()
    
    print("=" * 70)


def get_block_for_state(
    analysis: BlockAnalysis,
    basis_index: int
) -> Optional[Dict]:
    """
    Find which block contains a given basis state.
    
    Args:
        analysis: BlockAnalysis object
        basis_index: Index of the basis state (original basis order)
        
    Returns:
        Block info dictionary if found, None otherwise
    """
    for block_info in analysis.blocks:
        if basis_index in block_info['original_basis_indices']:
            # Find the position within the block
            pos_in_block = block_info['original_basis_indices'].index(basis_index)
            # Create a copy with position info
            result = block_info.copy()
            result['position_in_block'] = pos_in_block
            return result
    return None


def get_blocks_summary(analysis: BlockAnalysis) -> Dict:
    """
    Get a summary of block structure.
    
    Args:
        analysis: BlockAnalysis object
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_blocks': analysis.total_blocks,
        'total_dimension': analysis.total_dimension,
        'block_dimensions': analysis.block_dimensions,
        'largest_block': max(analysis.block_dimensions) if analysis.block_dimensions else 0,
        'smallest_block': min(analysis.block_dimensions) if analysis.block_dimensions else 0,
        'average_block_size': sum(analysis.block_dimensions) / len(analysis.block_dimensions) if analysis.block_dimensions else 0
    }
