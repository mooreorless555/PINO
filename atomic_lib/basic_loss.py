import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# Import necessary functions and constants from math_utils
from atomic_lib.math_utils import (
    compute_pos_dots,
    dist_to_point,
    dist_to_point_squared,
    calculate_angle_in_degrees,
    equal,
    less,
    less_than,
    greater_than,
    operation_or,
    dimX,
    dimZ,
    dimXZ,
    get_joint,
    PELVIS,
    LEFT_FOOT,
    RIGHT_FOOT
)
from utils.utils import MotionNormalizerTorch

# --- Constants ---
_NUM_JOINTS: int = 22
_JOINT_DIM: int = 3
_MOTION_FEATURE: int = 262

# --- Small helpers ---

def get_length(joints: torch.Tensor) -> int:
    """Gets the number of frames in a motion sequence."""
    return joints.shape[0]


def get_device(joints: torch.Tensor) -> torch.device:
    """Gets the device on which a tensor is located."""
    return joints.device


def _unpack_joints_from_motion(motion_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks motion data of shape (frames, -1, features) into joint data for two persons.
    Assumes first 22*3 dims are joints (T, 22, 3).
    """
    joints_list: List[torch.Tensor] = []
    for j in range(motion_data.shape[1]):  # person axis (2)
        motion_output = motion_data[:, j]
        joints3d = motion_output[:, :_NUM_JOINTS * _JOINT_DIM].reshape(-1, _NUM_JOINTS, _JOINT_DIM)
        joints_list.append(joints3d)
    return joints_list


def _denorm_and_unpack(motion_both: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denormalize (B/T, person_num, F) motion tensor and unpack into two persons' joints.
    """
    normalizer = MotionNormalizerTorch()
    motion_output_both = normalizer.backward(motion_both.to(device))
    return _unpack_joints_from_motion(motion_output_both)


def motion_prepare(data: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares motion from a tensor by denormalizing and converting it into joint data for two persons.
    """
    motion_both = data[0].reshape(data[0].shape[0], -1, _MOTION_FEATURE)
    return _denorm_and_unpack(motion_both, device)


def motion_prepare_from_path(motion_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads and prepares motion from a file path.
    """
    data = torch.load(motion_path)
    motion_tensor = data[0]
    motion_both = motion_tensor.reshape(motion_tensor.shape[0], -1, _MOTION_FEATURE)
    return _denorm_and_unpack(motion_both, device)


# --- Loss Functions (two-person direction wrapper) ---

def pair_direction_loss(
    joints1: torch.Tensor,
    joints2: torch.Tensor,
    gt_dir_dot: Optional[float] = None,
    gt_rdot1: Optional[float] = None,
    gt_rdot2: Optional[float] = None,
    target_pos1: Optional[torch.Tensor] = None,
    target_pos2: Optional[torch.Tensor] = None,
    start_t: Optional[int] = None,
    end_t: Optional[int] = None,
    eval_flag: bool = False
) -> torch.Tensor:
    """
    Calculates losses related to the orientation and relative direction of two persons.
    Uses the vector alignment loss (direction_loss) internally for facing targets.
    """
    device = get_device(joints1)
    time_slice = slice(start_t, end_t) if (start_t is not None and end_t is not None) else slice(None)
    dir_dot, r_dot1, r_dot2, n_vec1, n_vec2 = compute_pos_dots(joints1[time_slice], joints2[time_slice])

    loss = torch.tensor(0.0, device=device)
    count = 0

    if gt_dir_dot is not None:
        loss = loss + equal(dir_dot, torch.as_tensor(gt_dir_dot, device=device))
        count += 1
    if gt_rdot1 is not None:
        loss = loss + equal(r_dot1, torch.as_tensor(gt_rdot1, device=device))
        count += 1
    if gt_rdot2 is not None:
        loss = loss + equal(r_dot2, torch.as_tensor(gt_rdot2, device=device))
        count += 1

    # Person 1 facing a target direction
    if target_pos1 is not None:
        gt_vec = torch.tensor([0.0, -1.0], device=device)  # Negative Z-axis
        if eval_flag:
            direction_loss(n_vec1[:, :, [0, 2]], gt_vec, gt_dim=0, eval_flag=True)
        else:
            loss = loss + direction_loss(n_vec1[:, :, [0, 2]], gt_vec, gt_dim=0)
        count += 1

    # Person 2 facing a target direction
    if target_pos2 is not None:
        gt_vec = torch.tensor([0.0, -1.0], device=device)  # Negative Z-axis
        if eval_flag:
            direction_loss(n_vec2[:, :, [0, 2]], gt_vec, gt_dim=0, eval_flag=True)
        else:
            loss = loss + direction_loss(n_vec2[:, :, [0, 2]], gt_vec, gt_dim=0)
        count += 1

    return (loss / count) if count > 0 else torch.tensor(0.0, device=device)

def direction_loss(
    pred_vec: torch.Tensor,
    gt_vec: torch.Tensor,
    margin: float = 0.0,
    gt_dim: int = -1,
    eval_flag: bool = False
) -> torch.Tensor:
    """
    Loss to encourage a predicted vector to align with a ground truth vector.
    Corresponds to the Orientation Penalty.
    """
    pred_vec_norm = F.normalize(pred_vec, dim=-1)
    gt_vec_norm = F.normalize(gt_vec, dim=gt_dim)

    if eval_flag:
        angles = calculate_angle_in_degrees(pred_vec_norm, gt_vec_norm.unsqueeze(0))
        print(f"Angle: {angles.mean().item():.2f} degrees")
        return torch.tensor(0.0, device=pred_vec.device)

    # Penalize when cosine similarity is below threshold (e.g., 0.9)
    cosine_similarity = torch.sum(pred_vec_norm * gt_vec_norm, dim=-1)
    loss = torch.clamp(0.9 - cosine_similarity, min=0.0)
    return loss.mean()


def velocity_acceleration_loss(
    joints1: torch.Tensor,
    joints2: torch.Tensor,
    start_t: int,
    end_t: int,
    gt_velocity1: Optional[float] = None,
    gt_velocity2: Optional[float] = None,
    gt_acceleration1: Optional[float] = None,
    gt_acceleration2: Optional[float] = None,
    margin: float = 1e-6
) -> torch.Tensor:
    """Calculates losses related to motion velocity and acceleration."""
    time_slice = slice(start_t, end_t)

    # Person 1
    velocity1 = joints1[1:] - joints1[:-1]
    loss = torch.tensor(0.0, device=joints1.device)
    if gt_velocity1 is not None:
        loss = loss + less(velocity1[time_slice], torch.as_tensor(gt_velocity1, device=joints1.device), margin, sum_flag=True)
    if gt_acceleration1 is not None:
        acceleration1 = velocity1[1:] - velocity1[:-1]
        loss = loss + less(acceleration1[time_slice], torch.as_tensor(gt_acceleration1, device=joints1.device), margin, sum_flag=True)

    # Person 2
    velocity2 = joints2[1:] - joints2[:-1]
    if gt_velocity2 is not None:
        loss = loss + less(velocity2[time_slice], torch.as_tensor(gt_velocity2, device=joints2.device), margin, sum_flag=True)
    if gt_acceleration2 is not None:
        acceleration2 = velocity2[1:] - velocity2[:-1]
        loss = loss + less(acceleration2[time_slice], torch.as_tensor(gt_acceleration2, device=joints2.device), margin, sum_flag=True)

    return loss


def motion_overlap_loss(
    joints1: torch.Tensor,
    joints2: torch.Tensor,
    motion_data: torch.Tensor = None,
    pivot_idx:int = 0,
    nfeats: int = 262, # Feature dimension for one person
    inter_motion_dist_min: float = 0.5,
    inter_motion_dist_max: float = 100.0,
    other_motion_distance: float = 1.5
) -> torch.Tensor:
    """
    Penalizes characters for being too close or too far apart; and against other motions.
    Corresponds to the Relative Position Penalty in the paper.
    """
    device = get_device(joints1)
    root_1 = dimXZ(get_joint(joints1, PELVIS))
    root_2 = dimXZ(get_joint(joints2, PELVIS))

    # Loss between the two newly generated people
    dist_1_2 = dist_to_point(root_1, root_2)
    loss = greater_than(dist_1_2, inter_motion_dist_min).mean() + \
           less_than(dist_1_2, inter_motion_dist_max).mean()

    # Against pre-existing motions (from the input tensor)
    if motion_data is not None:
        # Assuming generated_motions shape is (1, frames, num_existing_people * nfeats)
        
        joints = motion_prepare(motion_data, device)
        
        num_existing_people = len(joints)

        # Iterate through each pre-existing person
        for i in range(num_existing_people):
            if i == pivot_idx:
                continue
            
            existing_joints = joints[i]
            
            root_existing = dimXZ(get_joint(existing_joints, PELVIS))
            
            loss = loss + greater_than(dist_to_point(root_2, root_existing), other_motion_distance).mean()

    return loss


def calc_region_loss(
    joints: torch.Tensor,
    region_type: str = 'circle',
    inside: bool = True,
    center: Optional[torch.Tensor] = None,
    radius: float = 1.0,
    width: float = 1.0,
    height: float = 1.0
) -> torch.Tensor:
    """
    Calculates the loss for being inside or outside a specified region.
    Corresponds to the Movement Region Penalty in the paper.
    """
    device = get_device(joints)
    center = torch.as_tensor([0.0, 0.0], device=device, dtype=joints.dtype) if center is None else center.to(device)
    root_joints = get_joint(joints, PELVIS)

    if region_type == 'circle':
        xz_root = dimXZ(root_joints)
        return (distance_loss_less(xz_root, center, radius).mean()
                if inside else distance_loss_greater(xz_root, center, radius).mean())

    if region_type == 'rectangle':
        x_root = dimX(root_joints)
        z_root = dimZ(root_joints)
        half_w, half_h = width / 2.0, height / 2.0
        cx, cz = center[0], center[1]

        if inside:
            loss_x1 = less_than(x_root, cx + half_w).mean()
            loss_x2 = greater_than(x_root, cx - half_w).mean()
            loss_z1 = less_than(z_root, cz + half_h).mean()
            loss_z2 = greater_than(z_root, cz - half_h).mean()
            return loss_x1 + loss_x2 + loss_z1 + loss_z2
        else:
            left_bound, right_bound = cx - half_w, cx + half_w
            bottom_bound, top_bound = cz - half_h, cz + half_h

            is_left_of_box = greater_than(x_root, left_bound)
            is_right_of_box = less_than(x_root, right_bound)
            is_below_box = greater_than(z_root, bottom_bound)
            is_above_box = less_than(z_root, top_bound)

            return operation_or(
                is_left_of_box,
                operation_or(is_right_of_box, operation_or(is_below_box, is_above_box))
            ).mean()

    raise ValueError("Invalid region_type. Use 'circle' or 'rectangle'.")


def distance_loss(joints1: torch.Tensor, joints2: torch.Tensor, gt_dist: float, margin: float = 0.0) -> torch.Tensor:
    """
    Encourages two points to be close within margin (uses squared error with tolerance).
    Note: Current implementation ignores gt_dist, matching original behavior.
    """
    return less(joints1, joints2, margin)


def distance_loss_greater(joints1: torch.Tensor, joints2: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Encourages distance^2 between two points to be greater than margin^2."""
    return greater_than(dist_to_point_squared(joints1, joints2), margin * margin)


def distance_loss_less(joints1: torch.Tensor, joints2: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Encourages distance^2 between two points to be less than margin^2."""
    return less_than(dist_to_point_squared(joints1, joints2), margin * margin)




XZLike = Union[torch.Tensor, Tuple[float, float], List[float]]


def _to_tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert `x` to a torch.Tensor on `device` with `dtype` if it isn't one already."""
    return x.to(device=device, dtype=dtype) if isinstance(x, torch.Tensor) \
           else torch.tensor(x, device=device, dtype=dtype)


def _extract_root(
    joints: torch.Tensor,
    joint_index: int,
    use_xz: bool,
) -> torch.Tensor:
    """
    Extract the specified joint from `joints` and optionally project to XZ.
    Input shape can be (T, J, 3) or (1, J, 3); this returns shape (T_or_1, 1, 2_or_3).
    """
    root = get_joint(joints, joint_index)              # (..., 1, 3)
    return dimXZ(root) if use_xz else root             # (..., 1, 2 or 3)


def _as_target_pos(
    tgt: XZLike,
    device: torch.device,
    dtype: torch.dtype,
    which: str,              # "start" or "end"
    joint_index: int,
    use_xz: bool,
) -> torch.Tensor:
    """
    Normalize any supported target format to a canonical shape (1, 1, 2_or_3):
    - Motion (T, J, 3): pick start/end frame, extract `joint_index`, project to XZ if requested
    - Pose (J, 3) or (1, J, 3): extract `joint_index`, project to XZ if requested
    - Vector (..., 2/3): use directly (project XYZ→XZ if requested), then reshape to (1, 1, *)
    """
    t = _to_tensor(tgt, device, dtype)

    # Case A: Full motion (T, J, 3)
    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[-2] >= joint_index + 1:
        frame = 0 if which == "start" else -1
        t_frame = t[frame:frame + 1]                      # (1, J, 3)
        pos = _extract_root(t_frame, joint_index, use_xz) # (1, 1, 2/3)
        return pos

    # Case B: Single pose (J, 3) or (1, J, 3)
    if (t.ndim == 2 and t.shape[-1] == 3 and t.shape[-2] >= joint_index + 1) or \
       (t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] == 1):
        if t.ndim == 2:
            t = t.unsqueeze(0)                            # (1, J, 3)
        pos = _extract_root(t, joint_index, use_xz)       # (1, 1, 2/3)
        return pos

    # Case C: Already a vector (..., 2/3)
    if t.ndim >= 1 and t.shape[-1] in (2, 3):
        if use_xz and t.shape[-1] == 3:
            t = t[..., [0, 2]]                            # XYZ → XZ
        # reshape to (1, 1, 2/3)
        while t.ndim < 3:
            t = t.unsqueeze(0)
        return t

    raise ValueError(f"Unsupported target format for {which}: shape={tuple(t.shape)}")


def start_end_pos_loss(
    joints1: torch.Tensor,
    joints2: torch.Tensor,
    gt_start_pos1: Optional[XZLike] = None,
    gt_start_pos2: Optional[XZLike] = None,
    gt_end_pos1: Optional[XZLike] = None,
    gt_end_pos2: Optional[XZLike] = None,
    start_frame_hold_ratio: int = 0,
    end_frame_hold_ratio: int = 0,
    margin: float = 0.01,
    *,
    joint_index: int = PELVIS,
    use_xz: bool = True,
    _gt_are_motions: bool = False,   # internal: when True, gt_* are motions (T, J, 3) and start/end frames are auto-picked
) -> torch.Tensor:
    """
    Start/end position loss with flexible GT inputs.
    By default, positions are computed on the XZ plane (use_xz=True).
    GT arguments can be motion, pose, or (x,z)/(x,y,z) vectors.
    """
    device = joints1.device
    dtype = joints1.dtype
    length = joints1.shape[0]
    if length == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Extract predicted roots
    pred_root_1 = _extract_root(joints1, joint_index, use_xz)  # (T,1,2/3)
    pred_root_2 = _extract_root(joints2, joint_index, use_xz)  # (T,1,2/3)

    # Slices
    start_len = 1 if start_frame_hold_ratio == 0 else max(1, length // start_frame_hold_ratio)
    start_slice = slice(0, start_len)

    end_len = 1 if end_frame_hold_ratio == 0 else max(1, length // end_frame_hold_ratio)
    end_slice = slice(length - end_len, length)

    loss = torch.tensor(0.0, device=device, dtype=dtype)

    # --- Start targets ---
    if gt_start_pos1 is not None:
        t = _as_target_pos(gt_start_pos1, device, dtype, "start", joint_index, use_xz)
        loss = loss + distance_loss(pred_root_1[start_slice], t, 0.0, margin)

    if gt_start_pos2 is not None:
        t = _as_target_pos(gt_start_pos2, device, dtype, "start", joint_index, use_xz)
        loss = loss + distance_loss(pred_root_2[start_slice], t, 0.0, margin)

    # --- End targets ---
    if gt_end_pos1 is not None:
        t = _as_target_pos(gt_end_pos1, device, dtype, "end", joint_index, use_xz)
        loss = loss + distance_loss(pred_root_1[end_slice], t, 0.0, margin)

    if gt_end_pos2 is not None:
        t = _as_target_pos(gt_end_pos2, device, dtype, "end", joint_index, use_xz)
        loss = loss + distance_loss(pred_root_2[end_slice], t, 0.0, margin)

    return loss


def start_end_pos_loss_from_gt(
    joints1: torch.Tensor,
    joints2: torch.Tensor,
    gt_joints1: torch.Tensor,
    gt_joints2: torch.Tensor,
    start_frame_hold_ratio: int = 0,
    end_frame_hold_ratio: int = 0,
    margin: float = 0.01,
    joint_index: int = PELVIS,
    use_xz: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper:
    Pass predicted joints (XYZ) and GT motions (XYZ) directly.
    The function will extract the start/end frames at `joint_index` from the GTs
    and compute the position loss on XZ (by default).
    """
    return start_end_pos_loss(
        joints1, joints2,
        gt_start_pos1=gt_joints1,  # same GT motion is passed; start/end are resolved internally
        gt_start_pos2=gt_joints2,
        gt_end_pos1=gt_joints1,
        gt_end_pos2=gt_joints2,
        start_frame_hold_ratio=start_frame_hold_ratio,
        end_frame_hold_ratio=end_frame_hold_ratio,
        margin=margin,
        joint_index=joint_index,
        use_xz=use_xz,
        _gt_are_motions=True,
    )

