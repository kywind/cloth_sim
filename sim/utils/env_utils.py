from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton import ParticleFlags
try:
    from newton.utils import transform_twist
except ImportError:
    from newton.math import transform_twist


def xyzw_to_wxyz(pose: np.ndarray):
    if len(pose.shape) == 1 and pose.shape[0] >= 7:
        new_pose = pose.copy()
        new_pose[3] = pose[6]
        new_pose[4] = pose[3]
        new_pose[5] = pose[4]
        new_pose[6] = pose[5]
    elif len(pose.shape) == 1 and pose.shape[0] == 4:
        new_pose = pose.copy()
        new_pose[0] = pose[3]
        new_pose[1] = pose[0]
        new_pose[2] = pose[1]
        new_pose[3] = pose[2]
    elif pose.shape[1] >= 7:
        new_pose = pose.copy()
        new_pose[:, 3] = pose[:, 6]
        new_pose[:, 4] = pose[:, 3]
        new_pose[:, 5] = pose[:, 4]
        new_pose[:, 6] = pose[:, 5]
    elif pose.shape[1] == 4:
        new_pose = pose.copy()
        new_pose[:, 0] = pose[:, 3]
        new_pose[:, 1] = pose[:, 0]
        new_pose[:, 2] = pose[:, 1]
        new_pose[:, 3] = pose[:, 2]
    return new_pose

def wxyz_to_xyzw(pose: np.ndarray):
    if len(pose.shape) == 1 and pose.shape[0] >= 7:
        new_pose = pose.copy()
        new_pose[6] = pose[3]
        new_pose[3] = pose[4]
        new_pose[4] = pose[5]
        new_pose[5] = pose[6]
    elif len(pose.shape) == 1 and pose.shape[0] == 4:
        new_pose = pose.copy()
        new_pose[3] = pose[0]
        new_pose[0] = pose[1]
        new_pose[1] = pose[2]
        new_pose[2] = pose[3]
    elif pose.shape[1] >= 7:
        new_pose = pose.copy()
        new_pose[:, 6] = pose[:, 3]
        new_pose[:, 3] = pose[:, 4]
        new_pose[:, 4] = pose[:, 5]
        new_pose[:, 5] = pose[:, 6]
    elif pose.shape[1] == 4:
        new_pose = pose.copy()
        new_pose[:, 3] = pose[:, 0]
        new_pose[:, 0] = pose[:, 1]
        new_pose[:, 1] = pose[:, 2]
        new_pose[:, 2] = pose[:, 3]
    return new_pose


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_robot: int,
    target: wp.array(dtype=wp.transform),
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    robot_id = wp.tid()
    tf = body_q[bodies_per_robot * robot_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target[robot_id])
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target[robot_id])
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[robot_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def compute_body_out(
    body_qd: wp.array(dtype=wp.spatial_vector), 
    body_id: int,
    bodies_per_robot: int,
    body_offset: wp.transform,
    # outputs
    body_out: wp.array(dtype=wp.spatial_vector)
):
    robot_id = wp.tid()
    mv = transform_twist(body_offset, body_qd[bodies_per_robot * robot_id + body_id])
    body_out[robot_id] = mv


@wp.kernel
def compute_gripper_bbox(
    joint_q: wp.array(dtype=float),
    dof_per_robot: int,
    # constants for bbox in gripper frame
    x_min: float, x_max: float,
    z_min: float, z_max: float,
    # outputs
    bbox_min: wp.array(dtype=wp.vec3),
    bbox_max: wp.array(dtype=wp.vec3),
):
    rid = wp.tid()

    # last two dofs are gripper finger joints in your setup
    q_l = joint_q[rid * dof_per_robot + (dof_per_robot - 2)]
    q_r = joint_q[rid * dof_per_robot + (dof_per_robot - 1)]

    # match your python logic:
    # y_min = min(-q[-1], 0.0)
    # y_max = max(q[-2], 0.0)
    y_min = wp.min(-q_r, 0.0)
    y_max = wp.max(q_l, 0.0)

    bbox_min[rid] = wp.vec3(x_min, y_min, z_min)
    bbox_max[rid] = wp.vec3(x_max, y_max, z_max)


@wp.kernel
def mark_grasped_particles(
    particle_q: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    bodies_per_robot: int,
    ee_body_id: int,
    ee_offset: wp.transform,
    bbox_min: wp.array(dtype=wp.vec3),
    bbox_max: wp.array(dtype=wp.vec3),
    # in/out
    locked_mask: wp.array(dtype=wp.int32),
    locked_local_pos: wp.array(dtype=wp.vec3),
    locked_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    pid = tid % particle_q.shape[0]
    rid = tid // particle_q.shape[0]

    ee_tf = body_q[bodies_per_robot * rid + ee_body_id] * ee_offset
    ee_tf_inv = wp.transform_inverse(ee_tf)

    p_w = particle_q[pid]
    p_g = wp.transform_point(ee_tf_inv, p_w)  # particle in *current* EE frame

    bmin = bbox_min[rid]
    bmax = bbox_max[rid]

    inside = True
    if p_g[0] < bmin[0] or p_g[0] > bmax[0]:
        inside = False
    if p_g[1] < bmin[1] or p_g[1] > bmax[1]:
        inside = False
    if p_g[2] < bmin[2] or p_g[2] > bmax[2]:
        inside = False

    if inside:
        old = wp.atomic_cas(locked_mask, pid, 0, rid + 1)
        if old == 0:
            locked_local_pos[pid] = p_g  # cache local attachment point
            wp.atomic_add(locked_count, 0, 1)


@wp.kernel
def update_grasped_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_q_1: wp.array(dtype=wp.vec3),
    particle_qd_1: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    bodies_per_robot: int,
    ee_body_id: int,
    ee_offset: wp.transform,
    locked_mask: wp.array(dtype=wp.int32),
    locked_local_pos: wp.array(dtype=wp.vec3),
):
    pid = wp.tid()

    m = locked_mask[pid]
    if m == 0:
        return

    rid = m - 1
    ee_index = bodies_per_robot * rid + ee_body_id

    # current EE transform
    ee_tf = body_q[ee_index] * ee_offset

    # desired particle position in world
    p_local = locked_local_pos[pid]
    p_world = wp.transform_point(ee_tf, p_local)
    particle_q[pid] = p_world
    particle_q_1[pid] = p_world

    # EE twist at the offset frame (uses your helper)
    # returns spatial vector [vx,vy,vz, wx,wy,wz] (consistent with your compute_body_out usage)
    ee_twist = transform_twist(ee_offset, body_qd[ee_index])

    v = wp.vec3(ee_twist[0], ee_twist[1], ee_twist[2])
    w = wp.vec3(ee_twist[3], ee_twist[4], ee_twist[5])

    # r_world = rotation(ee_tf) * p_local
    q = wp.transform_get_rotation(ee_tf)
    r_world = wp.quat_rotate(q, p_local)

    # point velocity v + w x r
    v_point = v + wp.cross(w, r_world)
    particle_qd[pid] = v_point
    particle_qd_1[pid] = v_point

    # optionally: clear force to reduce solver fighting
    # particle_f[pid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def write_particle_flags(
    locked_mask: wp.array(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if locked_mask[tid] == 0:
        particle_flags[tid] = ParticleFlags.ACTIVE
    else:
        particle_flags[tid] = 0


def resolve_asset_path(path: str) -> Path:
    asset_path = Path(path)
    if asset_path.is_absolute():
        return asset_path
    return Path(__file__).parents[2] / asset_path


def quat_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Shepperd's method: 3x3 rotation matrix → xyzw quaternion."""
    trace = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def pose_to_pos_quat(pose) -> tuple[np.ndarray, np.ndarray]:
    if pose is None:
        return np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.array(pose, dtype=np.float32).reshape(4, 4)
    return mat[:3, 3], quat_from_matrix(mat[:3, :3])


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    x, y, z, w = quat
    qvec = np.array([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def combine_transforms(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Compose two [px, py, pz, qx, qy, qz, qw] transforms."""
    parent = np.asarray(parent, dtype=np.float32)
    child = np.asarray(child, dtype=np.float32)
    p1, q1 = parent[:3], parent[3:7]
    p2, q2 = child[:3], child[3:7]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    q = np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)
    p = p1 + quat_rotate(q1, p2)
    return np.concatenate([p, q])


def axis_angle_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors (N, 3) to unit quaternions (N, 4) in xyzw order."""
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    safe = np.where(theta > 1e-8, theta, np.ones_like(theta))
    axis = aa / safe
    half = theta * 0.5
    xyz = axis * np.sin(half)
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=1)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batch quaternion product q1 * q2, both (N, 4) xyzw, returns (N, 4) xyzw."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=1)


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Convert a quaternion to a vec4."""
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d(dtype=wp.float32),
    gripper_value: float,
    num_arm_joints: int,
    num_gripper_joints: int,
    robot_id: int,
    joint_targets: wp.array(dtype=wp.float32),
):
    world_idx = wp.tid()
    num_total_joints = num_arm_joints + num_gripper_joints
    for j in range(num_arm_joints):
        joint_targets[robot_id * num_total_joints + j] = ik_solution[0, j]
    for j in range(num_gripper_joints):
        joint_targets[robot_id * num_total_joints + num_arm_joints + j] = gripper_value


def onehot(i, out_dim):
    """Generate a one-hot vector of length out_dim with a 1.0 at index i."""
    return wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)


@wp.kernel
def write_targets_kernel(
    targets: wp.array2d(dtype=float),
    out: wp.array(dtype=float),
    stride: int,
):
    env_id, joint_id = wp.tid()
    out[env_id * stride + joint_id] = targets[env_id, joint_id]


@wp.kernel
def scatter_ik_solutions_kernel(
    ik_solution: wp.array2d(dtype=wp.float32),   # (num_envs, n_coords)
    gripper_values: wp.array(dtype=wp.float32),  # (num_envs,)
    num_arm_joints: int,
    num_gripper_joints: int,
    robot_id: int,
    dofs_per_world: int,
    joint_targets: wp.array(dtype=wp.float32),
):
    world_idx = wp.tid()
    num_total_joints = num_arm_joints + num_gripper_joints
    base = world_idx * dofs_per_world + robot_id * num_total_joints
    for j in range(num_arm_joints):
        joint_targets[base + j] = ik_solution[world_idx, j]
    for j in range(num_gripper_joints):
        joint_targets[base + num_arm_joints + j] = gripper_values[world_idx]
