"""G1-29dof settings."""

from typing import Dict, List

from legged_gym import LEGGED_GYM_ROOT_DIR

# --- Number of DOFs ---
G1_NUM_DOF: int = 29

# --- Joint ordering (29-DOF) ---
G1_DOF_NAMES: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_DOF_INDEX: Dict[str, int] = {name: i for i, name in enumerate(G1_DOF_NAMES)}

# --- Body/link name groups (robot-specific) ---
G1_KEY_BODIES: List[str] = [
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
    "head_mocap",
]

G1_FEET_BODIES: List[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_HAND_BODIES: List[str] = ["left_rubber_hand", "right_rubber_hand"]
# --- Link names used for indexing ---
G1_TORSO_NAME: str = "pelvis"

# --- Default joint angles (29-DOF) ---
G1_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    "left_hip_pitch_joint": -0.2,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.4,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.2,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.4,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.2,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# --- Armature (29-DOF) ---
G1_DOF_ARMATURE: List[float] = (
    [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2
    + [0.0103] * 3
    + [0.003597] * 8
    + [0.00425] * 6
)

# --- Common DOF index groups ---
G1_ANKLE_DOF_IDX: List[int] = [
    G1_DOF_INDEX["left_ankle_pitch_joint"],
    G1_DOF_INDEX["left_ankle_roll_joint"],
    G1_DOF_INDEX["right_ankle_pitch_joint"],
    G1_DOF_INDEX["right_ankle_roll_joint"],
]

G1_WAIST_DOF_IDX: List[int] = [
    G1_DOF_INDEX["waist_roll_joint"],
    G1_DOF_INDEX["waist_pitch_joint"],
]


# --- Utility checks ---

def _get_last_dim(x):
    try:
        return x.shape[-1]
    except Exception:
        return None


def require_dof_dim(x, expected: int = G1_NUM_DOF, name: str = "dof_pos"):
    """Raise if x does not have the expected last-dimension size."""
    dim = _get_last_dim(x)
    if dim != expected:
        raise ValueError(f"{name} last dim mismatch: expected {expected}, got {dim}")


# --- Asset paths ---
G1_URDF_PATH: str = f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_29dof_custom.urdf"
G1_XML_PATH: str = f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_29dof_rev_1_0.xml"

__all__ = [
    "G1_URDF_PATH",
    "G1_XML_PATH",
    "G1_NUM_DOF",
    "G1_DOF_NAMES",
    "G1_DOF_INDEX",
    "G1_ANKLE_DOF_IDX",
    "G1_WAIST_DOF_IDX",
    "G1_KEY_BODIES",
    "G1_FEET_BODIES",
    "G1_HAND_BODIES",
    "G1_TORSO_NAME",
    "G1_DEFAULT_JOINT_ANGLES",
    "G1_DOF_ARMATURE",
    "require_dof_dim",
]
