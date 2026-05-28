"""生成式AI晶体结构发现模块 — 扩散模型 + CGCNN筛选 + DFT对接"""

from .crystal_representation import (
    CrystalStructure, ATOMIC_NUM_TO_ELEM, ELEM_TO_ATOMIC_NUM,
    _get_atom_feature, MAX_ATOMIC_NUMBER,
)
from .space_group_utils import (
    SPACE_GROUP_NAMES, CRYSTAL_SYSTEMS,
    get_crystal_system, get_lattice_constraints, apply_lattice_constraints,
    generate_random_structure, get_wyckoff_multiplicity,
    get_symmetry_equivalent_positions, reduce_to_asymmetric_unit,
)
from .egnn import EGNN, EGNNLayer, build_crystal_graph
from .diffusion_model import (
    CrystalDiffusion, NoiseScheduler, composition_to_vector,
)
from .cgcnn_proxy import CGCNNProxy, DefaultEnergyPredictor
from .validity_checker import StructureValidator, ValidityReport
from .structure_generator import (
    CrystalGenerator, prepare_dft_batch, generate_and_export,
)
