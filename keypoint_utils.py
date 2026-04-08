import numpy as np


COORDS_PER_JOINT = 3
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6


def robust_shoulder_scale(selected_3d: np.ndarray,
                          left_shoulder_idx: int = LEFT_SHOULDER_IDX,
                          right_shoulder_idx: int = RIGHT_SHOULDER_IDX,
                          min_scale: float = 1e-2) -> float:
    """Estimate signer scale from shoulder distance on non-empty frames."""
    if selected_3d.ndim != 3 or selected_3d.shape[-1] != 3:
        raise ValueError(f"Expected (T, J, 3) array, got {selected_3d.shape}")

    valid = np.linalg.norm(selected_3d.reshape(selected_3d.shape[0], -1), axis=1) > 0
    if not np.any(valid):
        return 1.0

    shoulders = selected_3d[valid]
    distances = np.linalg.norm(
        shoulders[:, left_shoulder_idx, :] - shoulders[:, right_shoulder_idx, :],
        axis=-1,
    )
    distances = distances[np.isfinite(distances) & (distances > min_scale)]
    if distances.size == 0:
        return 1.0
    return float(max(np.median(distances), min_scale))


def normalize_selected_keypoints(selected_3d: np.ndarray) -> np.ndarray:
    """Scale-normalize a (T, 75, 3)-style sequence using shoulder distance."""
    scale = robust_shoulder_scale(selected_3d)
    selected_3d = selected_3d / scale
    return np.nan_to_num(selected_3d, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_flat_keypoints(flat: np.ndarray, coords_per_joint: int = COORDS_PER_JOINT) -> np.ndarray:
    """Scale-normalize flattened keypoints with layout (T, J*3)."""
    if flat.ndim != 2 or flat.shape[1] % coords_per_joint != 0:
        raise ValueError(f"Expected (T, J*{coords_per_joint}) array, got {flat.shape}")
    shaped = flat.reshape(flat.shape[0], flat.shape[1] // coords_per_joint, coords_per_joint)
    shaped = normalize_selected_keypoints(shaped)
    return shaped.reshape(flat.shape[0], -1).astype(np.float32)
