import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# --- Joint Index Constants ---
PELVIS = 0
LEFT_HIP = 1
RIGHT_HIP = 2
SPINE1 = 3
LEFT_KNEE = 4
RIGHT_KNEE = 5
SPINE2 = 6
LEFT_ANKLE = 7
RIGHT_ANKLE = 8
SPINE3 = 9
LEFT_FOOT = 10
RIGHT_FOOT = 11
NECK = 12
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
HEAD = 15
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21

HML_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
]

# --- Basic Joint/Dimension Utilities ---

def get_joint(joints: torch.Tensor, joint_idx: int) -> torch.Tensor:
    """Extracts data for a specific joint index from a motion tensor."""
    return joints[:, joint_idx:joint_idx + 1, :]


def dimX(joints: torch.Tensor) -> torch.Tensor:
    """Extracts the X-dimension from joint data."""
    return joints[..., 0:1]


def dimY(joints: torch.Tensor) -> torch.Tensor:
    """Extracts the Y-dimension from joint data."""
    return joints[..., 1:2]


def dimZ(joints: torch.Tensor) -> torch.Tensor:
    """Extracts the Z-dimension from joint data."""
    return joints[..., 2:3]


def dimXZ(joints: torch.Tensor) -> torch.Tensor:
    """Extracts the XZ-plane data from joint data."""
    return joints[..., [0, 2]]


# --- Geometric and Distance Computations ---

def dist_to_point(joints_1: torch.Tensor, joints_2: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculates the Euclidean distance between two sets of points."""
    return torch.linalg.norm(joints_1 - joints_2, dim=dim, keepdim=True)


def dist_to_point_squared(joints_1: torch.Tensor, joints_2: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculates the squared Euclidean distance, avoiding a costly square root operation."""
    return torch.sum((joints_1 - joints_2) ** 2, dim=dim, keepdim=True)


def compute_pos_dots(joints1: torch.Tensor, joints2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes body facing vectors (via shoulders) and relative vector between two persons,
    returning dir dot and relation dots.
    """
    p1 = get_joint(joints1, PELVIS)
    rs1 = get_joint(joints1, RIGHT_SHOULDER)
    ls1 = get_joint(joints1, LEFT_SHOULDER)

    p2 = get_joint(joints2, PELVIS)
    rs2 = get_joint(joints2, RIGHT_SHOULDER)
    ls2 = get_joint(joints2, LEFT_SHOULDER)

    # Midpoints (zero-out y to project to XZ-plane)
    m1 = (rs1 + ls1) / 2.0
    m1[..., 1] = 0.0
    m2 = (rs2 + ls2) / 2.0
    m2[..., 1] = 0.0

    # Facing normals via cross product
    n_vec1 = F.normalize(torch.linalg.cross(rs1 - m1, ls1 - m1, dim=-1), dim=-1)
    n_vec2 = F.normalize(torch.linalg.cross(rs2 - m2, ls2 - m2, dim=-1), dim=-1)

    # Relative vector p1 -> p2
    r_vec1 = F.normalize(p2 - p1, dim=-1)

    dir_dot = torch.sum(n_vec1 * n_vec2, dim=-1)
    r_dot1 = torch.sum(r_vec1 * n_vec1, dim=-1)    # p1->p2 vs p1 facing
    r_dot2 = torch.sum(-r_vec1 * n_vec2, dim=-1)   # p2->p1 vs p2 facing

    return dir_dot, r_dot1, r_dot2, n_vec1, n_vec2


# --- Relational Loss Primitives ---

def less_than(value: torch.Tensor, margin: float) -> torch.Tensor:
    """Penalizes values that are greater than a margin: ReLU(value - margin)."""
    return F.relu(value - margin)


def greater_than(value: torch.Tensor, margin: float) -> torch.Tensor:
    """Penalizes values that are less than a margin: ReLU(margin - value)."""
    return F.relu(margin - value)


def equal(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss to encourage two tensors to be equal."""
    return F.mse_loss(pred, gt)


def less(pred: torch.Tensor, gt: torch.Tensor, margin: float, sum_flag: bool = False) -> torch.Tensor:
    """Penalizes if the squared difference between two values exceeds a margin."""
    loss = F.relu(((pred - gt) ** 2) - margin)
    return loss.sum() if sum_flag else loss.mean()


def operation_or(loss1: torch.Tensor, loss2: torch.Tensor) -> torch.Tensor:
    """
    Returns the minimum of two losses.
    Useful when any one of multiple constraints can be satisfied.
    """
    return torch.minimum(loss1, loss2)


# --- Specific Utilities ---

def calculate_angle_in_degrees(vec1: torch.Tensor, vec2: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Calculates the angle between two vectors in degrees."""
    vec1_norm = F.normalize(vec1, dim=dim)
    vec2_norm = F.normalize(vec2, dim=dim)
    dot_product = torch.sum(vec1_norm * vec2_norm, dim=dim)
    angles_rad = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    return torch.rad2deg(angles_rad)
