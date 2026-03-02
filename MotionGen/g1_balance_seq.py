import dataclasses
from typing import Sequence, Tuple, Dict, Optional
import os
import tyro

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

from pyroki.collision import RobotCollision


@dataclasses.dataclass
class RobotGeometryConfig:
    """Robot specific geometric hacks and offsets."""

    foot_center_offset_x: float = 0.03
    canonical_pelvis_z: float = 0.7
    support_foot_height: float = 0.03


@dataclasses.dataclass
class CostWeightsConfig:
    """Weights for the JAXLS optimization."""

    # Phase 1 / Phase 2 Base weights
    support_pos: float = 60.0
    support_ori: float = 30.0
    floating_pos: float = 10.0
    floating_ori: float = 2.0
    pelvis_pos: Tuple[float, float, float] = (5.0, 5.0, 20.0)
    pelvis_ori: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    # Multipliers & Phase specific
    support_lock_multiplier: float = 10.0
    floating_ori_multiplier: float = 0.1
    com_tracking: float = 50.0
    stability_phase2: float = 200.0

    # Smoothness & Rest
    joint_smoothness: float = 5.0
    base_smoothness: float = 10.0
    arm_rest: float = 0.5


@dataclasses.dataclass
class OptimizationConfig:
    """Hyperparameters for the physical constraints."""

    opt_support_x: float = 0.06
    opt_support_y: float = 0.03
    opt_tolerance: float = 0.005
    val_support_x: float = 0.10
    val_support_y: float = 0.035
    val_tolerance: float = 0.005
    shift_ratio: float = 1 / 3


@dataclasses.dataclass
class SequenceConfig:
    """General Sequence Generation parameters."""

    num_steps: int = 60
    body_height_range: Optional[Tuple[float, float]] = None
    enable_collision_check: bool = True

    optimized_joint_indices: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) + (22,23,24,25,26,27,28,30,31,32,33,34,35,36)

# -----------------------------
# Utilities
# -----------------------------


@jdc.jit
def compute_center_of_mass_jax(
    link_positions: jnp.ndarray,
    link_masses: jnp.ndarray,
    link_com_offsets: jnp.ndarray,
) -> jnp.ndarray:
    world_com_positions = link_positions + link_com_offsets
    total_mass = jnp.sum(link_masses)
    weighted_position = jnp.sum(link_masses[:, None] * world_com_positions, axis=0)
    com_position = jnp.where(
        total_mass > 1e-8,
        weighted_position / total_mass,
        jnp.mean(link_positions, axis=0),
    )
    return com_position

from loguru import logger as log
def extract_mass_properties(
    robot: pk.Robot, urdf: yourdfpy.URDF
) -> tuple[jnp.ndarray, jnp.ndarray]:
    masses = []
    com_locals = []
    for link_name in robot.links.names:
        assert link_name in urdf.link_map, f"link {link_name} not in urdf"
        link = urdf.link_map[link_name]
        if link.inertial is not None and link.inertial.mass is not None:
            mass = float(link.inertial.mass)
            if link.inertial.origin is not None:
                com_local = onp.array(link.inertial.origin[:3, 3])
            else:
                com_local = onp.zeros(3)
        else:
            mass = 0.0
            com_local = onp.zeros(3)
        masses.append(mass)
        com_locals.append(com_local)
    return jnp.array(masses), jnp.array(com_locals)


@jdc.jit
def point_to_support_polygon_distance_2d(
    point: jnp.ndarray,
    support_center: jnp.ndarray,
    support_size_x: float = 0.15,
    support_size_y: float = 0.08,
) -> jnp.ndarray:
    local_point = point - support_center
    dx = jnp.maximum(0.0, jnp.abs(local_point[0]) - support_size_x)
    dy = jnp.maximum(0.0, jnp.abs(local_point[1]) - support_size_y)
    return jnp.sqrt(dx**2 + dy**2 + 1e-12)


# -----------------------------
# Per-timestep Costs
# -----------------------------


@jaxls.Cost.create_factory
def single_foot_stability_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    masses: jnp.ndarray,
    com_local: jnp.ndarray,
    target_world_poses: jaxlie.SE3,
    support_foot_idx: int,
    weight: jax.Array | float,
    tolerance: float,
    support_x: float,
    support_y: float,
    foot_offset_x: float,
	optimized_joint_indices: jnp.ndarray,
) -> jax.Array:
    joint_cfg = vals[joint_var]
    mask = jnp.ones_like(joint_cfg, dtype=bool)
    mask = mask.at[optimized_joint_indices].set(False)
    
    joint_cfg_stopped = jax.lax.stop_gradient(joint_cfg)
    joint_cfg_for_stability = jnp.where(mask, joint_cfg_stopped, joint_cfg)

    base_pose = vals[T_world_base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg_for_stability)
    Ts_world_link = base_pose @ jaxlie.SE3(Ts_base_link)
    link_positions = Ts_world_link.translation()
    link_com_world_offsets = jax.vmap(
        lambda pose, local_com: pose.apply(local_com) - pose.translation()
    )(Ts_world_link, com_local)
    com_position = compute_center_of_mass_jax(
        link_positions, masses, link_com_world_offsets
    )
    translations = target_world_poses.translation()
    support_foot_pos = translations[support_foot_idx]
    com_2d = com_position[:2]
    support_foot_2d = support_foot_pos[:2]

    # from jax import debug as jdebug
    # jdebug.print("support_foot_2d: {}", support_foot_2d)
    support_foot_2d = support_foot_2d.at[0].add(foot_offset_x)
    distance = point_to_support_polygon_distance_2d(
        com_2d, support_foot_2d, support_x, support_y
    )
    violation = jnp.maximum(0.0, distance - tolerance)
    return jnp.array([violation * weight])


@jaxls.Cost.create_factory
def com_tracking_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    masses: jnp.ndarray,
    com_local: jnp.ndarray,
    target_com_2d: jnp.ndarray,
    weight: float,
    optimized_joint_indices: jnp.ndarray,
) -> jax.Array:
    joint_cfg = vals[joint_var]
    

    mask = jnp.ones_like(joint_cfg, dtype=bool)
    mask = mask.at[optimized_joint_indices].set(False)
    
    joint_cfg_stopped = jax.lax.stop_gradient(joint_cfg)
    joint_cfg_for_com = jnp.where(mask, joint_cfg_stopped, joint_cfg)

    base_pose = vals[T_world_base_var]
    
    Ts_base_link = robot.forward_kinematics(joint_cfg_for_com) 
    Ts_world_link = base_pose @ jaxlie.SE3(Ts_base_link)
    
    link_positions = Ts_world_link.translation()
    link_com_world_offsets = jax.vmap(
        lambda pose, local_com: pose.apply(local_com) - pose.translation()
    )(Ts_world_link, com_local)
    com_position = compute_center_of_mass_jax(
        link_positions, masses, link_com_world_offsets
    )
    return (com_position[:2] - target_com_2d) * weight


@jaxls.Cost.create_factory
def base_height_range_cost(
    vals: jaxls.VarValues,
    T_world_base_var: jaxls.Var[jaxlie.SE3],
    min_height: float,
    max_height: float,
    weight: float,
) -> jax.Array:
    base_pose = vals[T_world_base_var]
    z = base_pose.translation()[-1]
    below = jnp.maximum(0.0, min_height - z)
    above = jnp.maximum(0.0, z - max_height)
    return jnp.array([(below + above) * weight])


@jaxls.Cost.create_factory
def base_smoothness_cost(
    vals: jaxls.VarValues,
    T_world_base_curr: jaxls.Var[jaxlie.SE3],
    T_world_base_prev: jaxls.Var[jaxlie.SE3],
    weight: float,
) -> jax.Array:
    residual = (vals[T_world_base_prev].inverse() @ vals[T_world_base_curr]).log()
    return (residual * weight).flatten()


# -----------------------------
# Helper functions
# -----------------------------


def get_link_world_pose_from_cfg(
    robot: pk.Robot,
    base_pose: jaxlie.SE3,
    joint_cfg: jnp.ndarray,
    link_index: int,
) -> jaxlie.SE3:
    Ts_base_link = robot.forward_kinematics(joint_cfg)
    T_world_links = base_pose @ jaxlie.SE3(Ts_base_link)
    wxyz_xyz = T_world_links.wxyz_xyz[link_index]
    wxyz = wxyz_xyz[:4]
    xyz = wxyz_xyz[4:]
    return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), xyz)


def slerp_quat(wxyz_a: jnp.ndarray, wxyz_b: jnp.ndarray, alpha: float) -> jnp.ndarray:
    # Simple normalized lerp as a fallback
    q = (1.0 - alpha) * wxyz_a + alpha * wxyz_b
    return q / jnp.maximum(1e-8, jnp.linalg.norm(q))


def interpolate_se3(T_a: jaxlie.SE3, T_b: jaxlie.SE3, alpha: float) -> jaxlie.SE3:
    pa = T_a.translation()
    pb = T_b.translation()
    p = (1.0 - alpha) * pa + alpha * pb
    qa = T_a.rotation().wxyz
    qb = T_b.rotation().wxyz
    q = slerp_quat(qa, qb, alpha)
    return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(q), p)


# -----------------------------
# Sequence Optimization
# -----------------------------


def optimize_motion_sequence(
    robot: pk.Robot,
    urdf: yourdfpy.URDF,
    support_foot_index: int,
    target_foot_pose: Tuple[Sequence[float], Sequence[float]],
	pelvis_height: float,
    config: SequenceConfig,
    weights: CostWeightsConfig,
    geometry: RobotGeometryConfig,
    opt_params: OptimizationConfig,
    init_cfg: jnp.ndarray,
    robot_collision: Optional[RobotCollision] = None,
    world_coll_list: Optional[Sequence[pk.collision.CollGeom]] = None,
    actuated_index: Optional[Sequence[int]] = None,
    upper_body_joints: Optional[Sequence[str]] = None,
) -> Tuple[onp.ndarray, bool]:
    num_steps = config.num_steps
    assert num_steps >= 2

    default_cfg = (
        jnp.asarray(init_cfg) if init_cfg is not None else robot.joints.default_cfg
    )

    base_pos0 = jnp.array([0.0, 0.0, 0.7])
    base_wxyz0 = jnp.array([1.0, 0.0, 0.0, 0.0])

    # Compute initial world poses of feet and pelvis
    base_pose0 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(base_wxyz0), base_pos0
    )
    left_idx = robot.links.names.index("left_ankle_roll_link")
    right_idx = robot.links.names.index("right_ankle_roll_link")
    pelvis_idx = robot.links.names.index("pelvis")

    T_left0 = get_link_world_pose_from_cfg(robot, base_pose0, default_cfg, left_idx)
    T_right0 = get_link_world_pose_from_cfg(robot, base_pose0, default_cfg, right_idx)
    T_pelvis0 = get_link_world_pose_from_cfg(robot, base_pose0, default_cfg, pelvis_idx)

    # Override initial link world poses with canonical init pose
    T_left0 = jaxlie.SE3.from_matrix(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    T_right0 = jaxlie.SE3.from_matrix(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    T_pelvis0 = jaxlie.SE3.from_matrix(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, geometry.canonical_pelvis_z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    # Supporting and floating selection
    if support_foot_index == 0:
        T_support0 = T_left0
        T_floating0 = T_right0
        floating_link_idx = right_idx
    else:
        T_support0 = T_right0
        T_floating0 = T_left0
        floating_link_idx = left_idx


    # Desired final floating foot pose (world)
    des_pos, des_wxyz = target_foot_pose
    T_floating_target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(jnp.array(des_wxyz)), jnp.array(des_pos)
    )

    pelvis_target_height = T_pelvis0.translation()[-1]
    if pelvis_height is not None:
        pelvis_target_height = jnp.array(pelvis_height)

    T_pelvis_target = jaxlie.SE3.from_rotation_and_translation(
        T_pelvis0.rotation(),
        T_pelvis0.translation().at[-1].set(pelvis_target_height),
    )

    print(f"T_pelvis_target: {T_pelvis_target}")

    # time steps
    ts = jnp.linspace(0.0, 1.0, num_steps)
    # init guess for optimization
    base_pos_guess = jnp.repeat(base_pos0[None, :], num_steps, axis=0)
    base_wxyz_guess = jnp.repeat(base_wxyz0[None, :], num_steps, axis=0)
    cfg_guess = jnp.repeat(default_cfg[None, :], num_steps, axis=0)

    # Create variables for all timesteps
    var_q = robot.joint_var_cls(jnp.arange(num_steps))
    var_base = jaxls.SE3Var(jnp.arange(num_steps))

    @jaxls.Cost.create_factory
    def init_lock_cost(
        vals: jaxls.VarValues,
        var_q0: jaxls.Var[jax.Array],
        var_base0: jaxls.Var[jaxlie.SE3],
        q0: jnp.ndarray,
        base0: jaxlie.SE3,
        w_q: float,
        w_base: float,
    ) -> jax.Array:
        res_q = (vals[var_q0] - q0) * w_q
        res_base = (base0.inverse() @ vals[var_base0]).log() * w_base
        return jnp.concatenate([res_q, res_base])

    # if support_foot_index == 0:
    # 	T_left_traj = [T_support0] * num_steps
    # 	T_right_traj = [interpolate_se3(T_floating0, T_floating_target, float(s)) for s in onp.array(ts)]
    # else:
    # 	T_left_traj = [interpolate_se3(T_floating0, T_floating_target, float(s)) for s in onp.array(ts)]
    # 	T_right_traj = [T_support0] * num_steps
    # T_pelvis_traj = [interpolate_se3(T_pelvis0, T_pelvis_target, float(s)) for s in onp.array(ts)]

    num_shift = max(2, int(num_steps * opt_params.shift_ratio))
    num_lift = num_steps - num_shift

    t_shift = jnp.linspace(0.0, 1.0, num_shift)
    t_lift = jnp.linspace(0.0, 1.0, num_lift)

    # Calculate canonical initial CoM (approx 0,0)
    com0 = jnp.array([0.0, 0.0])
    support_foot_2d = (
        T_support0.translation()[:2].at[0].add(geometry.foot_center_offset_x)
    )

    # Target CoM Sequence: Interpolate to support foot, then hold
    com_target_seq = jnp.vstack(
        [
            jnp.stack(
                [(1 - s) * com0 + s * support_foot_2d for s in onp.array(t_shift)]
            ),
            jnp.repeat(support_foot_2d[None, :], num_lift, axis=0),
        ]
    )

    # Shift pelvis over the support foot during Phase 1
    T_pelvis_shifted = jaxlie.SE3.from_rotation_and_translation(
        T_pelvis0.rotation(),
        jnp.array([support_foot_2d[0], support_foot_2d[1], T_pelvis0.translation()[2]]),
    )

    T_left_traj, T_right_traj, T_pelvis_traj = [], [], []

    if support_foot_index == 0:
        # Left Support, Right Floating
        for s in onp.array(t_shift):
            T_left_traj.append(T_support0)
            T_right_traj.append(T_floating0)  # Keep floating foot on ground
            T_pelvis_traj.append(interpolate_se3(T_pelvis0, T_pelvis_shifted, float(s)))
        for s in onp.array(t_lift):
            T_left_traj.append(T_support0)
            T_right_traj.append(
                interpolate_se3(T_floating0, T_floating_target, float(s))
            )
            T_pelvis_traj.append(
                interpolate_se3(T_pelvis_shifted, T_pelvis_target, float(s))
            )
    else:
        # Right Support, Left Floating
        for s in onp.array(t_shift):
            T_left_traj.append(T_floating0)  # Keep floating foot on ground
            T_right_traj.append(T_support0)
            T_pelvis_traj.append(interpolate_se3(T_pelvis0, T_pelvis_shifted, float(s)))
        for s in onp.array(t_lift):
            T_left_traj.append(
                interpolate_se3(T_floating0, T_floating_target, float(s))
            )
            T_right_traj.append(T_support0)
            T_pelvis_traj.append(
                interpolate_se3(T_pelvis_shifted, T_pelvis_target, float(s))
            )

    target_link_indices = jnp.array([left_idx, right_idx, pelvis_idx], dtype=jnp.int32)
    link_masses, link_com_local = extract_mass_properties(robot, urdf)
    link_com_local = jnp.zeros_like(link_com_local)

    # Dynamic weights based on support foot (from g1_balance_sample)
    # support_pos_weight = 60.0
    # support_ori_weight = 30.0
    # floating_pos_weight = 10.0
    # floating_ori_weight = 2.0
    support_pos_weight = weights.support_pos
    support_ori_weight = weights.support_ori
    floating_pos_weight = weights.floating_pos
    floating_ori_weight = weights.floating_ori
    left_support_pos_weight = jnp.array(
        [
            [
                support_pos_weight,
                support_pos_weight,
                support_pos_weight,
            ],  # left foot (support)
            [
                floating_pos_weight,
                floating_pos_weight,
                floating_pos_weight,
            ],  # right foot (floating)
            [5.0, 5.0, 20.0],  # pelvis
        ]
    )
    left_support_ori_weight = jnp.array(
        [
            [
                support_ori_weight,
                support_ori_weight,
                support_ori_weight,
            ],  # left foot (support)
            [
                floating_ori_weight,
                floating_ori_weight,
                floating_ori_weight,
            ],  # right foot (floating)
            [2.0, 2.0, 2.0],  # pelvis
        ]
    )
    right_support_pos_weight = jnp.array(
        [
            [
                floating_pos_weight,
                floating_pos_weight,
                floating_pos_weight,
            ],  # left foot (floating)
            [
                support_pos_weight,
                support_pos_weight,
                support_pos_weight,
            ],  # right foot (support)
            [5.0, 5.0, 20.0],  # pelvis
        ]
    )
    right_support_ori_weight = jnp.array(
        [
            [
                floating_ori_weight,
                floating_ori_weight,
                floating_ori_weight,
            ],  # left foot (floating)
            [
                support_ori_weight,
                support_ori_weight,
                support_ori_weight,
            ],  # right foot (support)
            [2.0, 2.0, 2.0],  # pelvis
        ]
    )
    pos_weight = jax.lax.select(
        support_foot_index == 0,
        left_support_pos_weight,
        right_support_pos_weight,
    )
    ori_weight = jax.lax.select(
        support_foot_index == 0,
        left_support_ori_weight,
        right_support_ori_weight,
    )

    support_row = 0 if support_foot_index == 0 else 1
    non_support_row = 1 if support_foot_index == 0 else 0
    pos_weight = pos_weight.at[support_row].set(pos_weight[support_row] * 10.0)
    ori_weight = ori_weight.at[support_row].set(ori_weight[support_row] * 10.0)
    # pos_weight = pos_weight.at[non_support_row].set(pos_weight[non_support_row] * 0.1)
    ori_weight = ori_weight.at[non_support_row].set(ori_weight[non_support_row] * 0.1)


    costs: list[jaxls.Cost] = []

    rot_seq = jnp.stack(
        [
            jnp.stack(
                [
                    T_left_traj[t].rotation().wxyz,
                    T_right_traj[t].rotation().wxyz,
                    T_pelvis_traj[t].rotation().wxyz,
                ]
            )
            for t in range(num_steps)
        ]
    )
    trans_seq = jnp.stack(
        [
            jnp.stack(
                [
                    T_left_traj[t].translation(),
                    T_right_traj[t].translation(),
                    T_pelvis_traj[t].translation(),
                ]
            )
            for t in range(num_steps)
        ]
    )
    T_world_targets_seq = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(rot_seq), trans_seq
    )

    # Batch static arguments for broadcasting across time
    robot_batched = jax.tree.map(lambda x: x[None], robot)
    pos_weight_b = jnp.broadcast_to(pos_weight, (num_steps, 3, 3))
    ori_weight_b = jnp.broadcast_to(ori_weight, (num_steps, 3, 3))
    target_link_indices_b = jnp.broadcast_to(
        target_link_indices[None, :], (num_steps, target_link_indices.shape[0])
    )
    link_masses_b = link_masses[None, :]
    link_com_local_b = link_com_local[None, :, :]

    support_row = 0 if support_foot_index == 0 else 1
    floating_row = 1 if support_foot_index == 0 else 0

    pos_weight_phase1 = pos_weight.at[floating_row].set(pos_weight[support_row])
    ori_weight_phase1 = ori_weight.at[floating_row].set(ori_weight[support_row])

    pos_weight_b = jnp.vstack(
        [
            jnp.repeat(pos_weight_phase1[None, :, :], num_shift, axis=0),
            jnp.repeat(
                pos_weight[None, :, :], num_lift, axis=0
            ),  # Original phase 2 weights
        ]
    )
    ori_weight_b = jnp.vstack(
        [
            jnp.repeat(ori_weight_phase1[None, :, :], num_shift, axis=0),
            jnp.repeat(ori_weight[None, :, :], num_lift, axis=0),
        ]
    )

    actuated_count = int(robot.joints.num_actuated_joints)
    actuated_names = [robot.joints.names[i] for i in actuated_index]
    per_joint_rest_list = []
    for name in actuated_names:
        if any(k in name for k in upper_body_joints):
            per_joint_rest_list.append(weights.arm_rest if weights.arm_rest is not None else 0.)
        else:
            per_joint_rest_list.append(0.0)

    per_joint_rest = jnp.array(per_joint_rest_list)
    per_joint_rest_b = jnp.broadcast_to(
        per_joint_rest[None, :], (num_steps, actuated_count)
    )

    costs_stage1: list[jaxls.Cost] = []
    costs_stage1.append(
        pk.costs.pose_cost_with_base(
            robot=robot_batched,
            joint_var=var_q,
            T_world_base_var=var_base,
            target_pose=T_world_targets_seq,
            target_link_indices=target_link_indices_b,
            pos_weight=pos_weight_b,
            ori_weight=ori_weight_b,
        )
    )
    costs_stage1.append(pk.costs.limit_cost(robot=robot_batched, joint_var=var_q, weight=500.0))
    costs_stage1.append(
        pk.costs.rest_cost(
            joint_var=var_q,
            rest_pose=jnp.repeat(var_q.default_factory()[None, :], num_steps, axis=0),
            weight=per_joint_rest_b,
        )
    )
    costs_stage1.append(
        pk.costs.smoothness_cost(
            curr_joint_var=robot.joint_var_cls(jnp.arange(1, num_steps)),
            past_joint_var=robot.joint_var_cls(jnp.arange(0, num_steps - 1)),
            weight=jnp.array([5.0]),
        )
    )
    costs_stage1.append(
        base_smoothness_cost(
            T_world_base_curr=jaxls.SE3Var(jnp.arange(1, num_steps)),
            T_world_base_prev=jaxls.SE3Var(jnp.arange(0, num_steps - 1)),
            weight=jnp.array([10.0]),
        )
    )
    costs_stage1.append(
        init_lock_cost(
            var_q0=robot.joint_var_cls(0),
            var_base0=jaxls.SE3Var(0),
            q0=default_cfg,
            base0=base_pose0,
            w_q=0.1,
            w_base=5.0,
        )
    )
    if config.enable_collision_check:
        robot_coll_model = (
            robot_collision
            if robot_collision is not None
            else RobotCollision.from_urdf(urdf, ignore_immediate_adjacents=True)
        )
        robot_coll_batched = jax.tree.map(
            lambda x: x[None] if hasattr(x, "shape") else x,
            robot_coll_model,
        )
        # costs_stage1.append(
        # 	pk.costs.self_collision_cost(
        # 		robot_batched,
        # 		robot_coll_batched,
        # 		var_q,
        # 		margin=-0.05,
        # 		weight=0.002,
        # 	)
        # )

        costs.extend(
            [
                pk.costs.world_collision_cost(
                    robot_batched, robot_coll_batched, var_q, world_coll, 0.05, 500.0
                )
                for world_coll in world_coll_list
            ]
        )

    from loguru import logger as log
    import time

    st = time.time()
    problem_stage1 = jaxls.LeastSquaresProblem(
        costs_stage1, [var_q, var_base]
    ).analyze()
    solution_stage1 = problem_stage1.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_q.with_value(cfg_guess),
                var_base.with_value(
                    jax.vmap(
                        lambda wxyz, pos: jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(wxyz), pos
                        )
                    )(base_wxyz_guess, base_pos_guess)
                ),
            ]
        ),
        verbose=False,
        linear_solver="dense_cholesky",
    )
    log.info(f"Time taken to analyze problem_stage1: {time.time() - st}")

    st = time.time()
    costs_stage2: list[jaxls.Cost] = list(costs_stage1)
    costs_stage2.append(
        com_tracking_cost(
            robot_batched,
            var_q,
            var_base,
            link_masses_b,
            link_com_local_b,
            com_target_seq,
            weight=50.0,
            optimized_joint_indices=jnp.array(config.optimized_joint_indices)[None, :],
        )
    )

    stability_weight_b = jnp.concatenate(
        [jnp.zeros(num_shift), jnp.full(num_lift, weights.stability_phase2)]
    )
    costs_stage2.append(
        single_foot_stability_cost(
            robot_batched,
            var_q,
            var_base,
            link_masses_b,
            link_com_local_b,
            T_world_targets_seq,
            support_foot_index,
            weight=stability_weight_b,
            tolerance=opt_params.opt_tolerance,
            support_x=opt_params.opt_support_x,
            support_y=opt_params.opt_support_y,
            foot_offset_x=geometry.foot_center_offset_x,
			optimized_joint_indices=jnp.array(config.optimized_joint_indices)[None, :],
        )
    )
    if num_steps >= 5:
        costs_stage2.append(
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, num_steps - 2)),
                robot.joint_var_cls(jnp.arange(4, num_steps)),
                robot.joint_var_cls(jnp.arange(3, num_steps - 1)),
                robot.joint_var_cls(jnp.arange(1, num_steps - 3)),
                robot.joint_var_cls(jnp.arange(0, num_steps - 4)),
                dt=1.0 / 30.0,
                weight=0.02,
            )
        )

    problem_stage2 = jaxls.LeastSquaresProblem(
        costs_stage2, [var_q, var_base]
    ).analyze()
    solution, summary = problem_stage2.solve(
        initial_vals=solution_stage1,
        verbose=True,
        linear_solver="dense_cholesky",
        return_summary=True,
    )
    had_nans = bool(onp.isnan(onp.array(summary.termination_deltas)).any())
    log.info(f"Time taken to analyze problem_stage2: {time.time() - st}")

    # Extract results
    base_seq = solution[var_base]
    q_seq = solution[var_q]
    qpos_seq = []
    for t in range(num_steps):
        pos = onp.array(base_seq.wxyz_xyz[t][4:])
        wxyz = onp.array(base_seq.wxyz_xyz[t][:4])
        cfg = onp.array(q_seq[t])
        qpos_seq.append(onp.concatenate([pos, wxyz, cfg]))
    return onp.stack(qpos_seq), had_nans



# -----------------------------
# Stability Validation
# -----------------------------


def validate_motion_stability(
    robot: pk.Robot,
    urdf: yourdfpy.URDF,
    qpos_seq: onp.ndarray,
    support_foot_index: int,
    geometry: RobotGeometryConfig,
    opt_params: OptimizationConfig,
) -> bool:
    """
    Validates if the Center of Mass (CoM) remains within the support polygon
    for all evaluated frames in a motion sequence.
    """
    # Extract mass properties
    masses, com_local = extract_mass_properties(robot, urdf)
    com_local = jnp.zeros_like(com_local) 
    
    # Determine the support foot link index
    support_foot_name = (
        "left_ankle_roll_link" if support_foot_index == 0 else "right_ankle_roll_link"
    )
    support_link_idx = robot.links.names.index(support_foot_name)

    num_steps = qpos_seq.shape[0]
    prefix_length = max(2, int(num_steps * opt_params.shift_ratio))

    for t in range(prefix_length, num_steps):
        pos = qpos_seq[t][:3]
        wxyz = qpos_seq[t][3:7]
        cfg = qpos_seq[t][7:]

        # Construct base pose
        base_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos)

        # Compute forward kinematics
        Ts_base_link = robot.forward_kinematics(cfg)
        Ts_world_link = base_pose @ jaxlie.SE3(Ts_base_link)

        # Compute Center of Mass (CoM) position
        link_positions = Ts_world_link.translation()
        link_com_world_offsets = jax.vmap(
            lambda pose, local_com: pose.apply(local_com) - pose.translation()
        )(Ts_world_link, com_local)
        
        com_position = compute_center_of_mass_jax(
            link_positions, masses, link_com_world_offsets
        )

        # Get support foot position using helper
        support_foot_pose = get_link_world_pose_from_cfg(
            robot, base_pose, cfg, support_link_idx
        )
        support_foot_pos = support_foot_pose.translation()

        # Compute 2D distance from CoM to support polygon
        com_2d = com_position[:2]
        support_foot_2d = support_foot_pos[:2]
        support_foot_2d = support_foot_2d.at[0].add(geometry.foot_center_offset_x)
        
        distance = point_to_support_polygon_distance_2d(
            com_2d, support_foot_2d, opt_params.val_support_x, opt_params.val_support_y
        )

        if distance > opt_params.val_tolerance:
            print(
                f"Frame {t}: COM distance {distance:.6f} exceeds tolerance {opt_params.val_tolerance:.6f}"
            )
            print(f"  COM 2D: {com_2d}")
            print(f"  Support foot 2D: {support_foot_2d}")
            return False

    return True