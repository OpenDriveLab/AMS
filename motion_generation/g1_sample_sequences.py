#!/usr/bin/env python3
import os
import pickle
import logging
import dataclasses
from typing import Mapping, Tuple, Optional
from pathlib import Path
from datetime import datetime

import tyro
import joblib
import numpy as np
import jax
import yourdfpy
import jax.numpy as jnp

from g1_balance_seq import (
    OptimizationConfig,
    CostWeightsConfig,
    RobotGeometryConfig,
    SequenceConfig,
    optimize_motion_sequence,
    validate_motion_stability,
)

import logging
import pyroki as pk
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as sRot


@dataclasses.dataclass
class SamplingConfig:
    num_samples: int = 4
    side: str = "both"
    urdf_path: str = "robot/unitree_description/urdf/g1/g1_sysid_29dof.urdf"
    output_path: str = "motions/g1/sampled_static_poses"
    log_file: str = "sampling.log"
    max_retries: int = 20

    # Timing
    stand_frames: int = 30
    fps: int = 30
    preview_seconds: float = 0.0

    default_joints: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
            "left_knee_joint": 0.3,
            "right_knee_joint": 0.3,
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": -0.0,
            "left_shoulder_roll_joint": 1.5,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.5,
            "right_shoulder_pitch_joint": -0.0,
            "right_shoulder_roll_joint": -1.5,
            "right_shoulder_yaw_joint": -0.0,
            "right_elbow_joint": 1.5,
        }
    )

    seed: int = 0
    randomize_init: bool = True

    random_init_ranges: Mapping[str, Tuple[float, float]] = dataclasses.field(
		default_factory=lambda: {
			"waist_yaw_joint": (-0.3, 0.3),
			"waist_roll_joint": (-0.2, 0.2),
			"waist_pitch_joint": (-0.2, 0.2),
			"left_shoulder_pitch_joint": (-0.5, 0.5),
			"left_shoulder_roll_joint": (0.1, 1.2),
			"left_shoulder_yaw_joint": (-0.2, 0.8),
			"left_elbow_joint": (-0.6, 0.6),
			"right_shoulder_pitch_joint": (-0.5, 0.5),
			"right_shoulder_roll_joint": (-1.2, 0.1),
			"right_shoulder_yaw_joint": (-0.8, 0.2),
			"right_elbow_joint": (-0.6, 0.6),
		}
	)
    
    upper_body_joints = [
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
    
    # actuated_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,30,31,32,33]
    actuated_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,30,31,32,33,34,35,36]

    float_x_range: Tuple[float, float] = (-0.4, 0.6)
    float_y_abs_range: Tuple[float, float] = (0.1, 0.4)
    float_z_range: Tuple[float, float] = (0.1, 1.0)
    pelvis_height_range: Tuple[float, float] = (0.30, 0.8)


@dataclasses.dataclass
class AppConfig:
    """Main Application Configuration."""

    sample: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    seq: SequenceConfig = dataclasses.field(default_factory=SequenceConfig)
    opt: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)
    weights: CostWeightsConfig = dataclasses.field(default_factory=CostWeightsConfig)
    geom: RobotGeometryConfig = dataclasses.field(default_factory=RobotGeometryConfig)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def wxyz_to_axis_angle(q_wxyz: np.ndarray) -> np.ndarray:
    # Normalize
    q = q_wxyz.astype(np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = np.array([x, y, z]) / s
    return (axis * angle).astype(np.float32)


def slerp_quat_wxyz(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    # Both wxyz
    a = a / max(1e-12, np.linalg.norm(a))
    b = b / max(1e-12, np.linalg.norm(b))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        res = a + t * (b - a)
        return res / max(1e-12, np.linalg.norm(res))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta0 = np.sin(theta0)
    theta = theta0 * t
    s = np.sin(theta)
    s0 = np.cos(theta) - dot * s / sin_theta0
    s1 = s / sin_theta0
    return s0 * a + s1 * b


def sample_floating_foot_pose(
    support_is_left: bool,
    x_range: Tuple[float, float],
    y_abs_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    # x forward, y lateral sign opposite of support leg
    x = np.random.uniform(*x_range)
    y_mag = np.random.uniform(*y_abs_range)
    y = -y_mag if support_is_left else y_mag
    z = np.random.uniform(*z_range)
    pos = np.array([x, y, z], dtype=np.float32)
    wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return pos, wxyz



def sample_init_cfg_actuated_order(
    robot: pk.Robot,
    fixed_values: Mapping[str, float],
    uniform_ranges: Mapping[str, Tuple[float, float]],
    rng: np.random.Generator,
    default_mode: str = "zero",  # "zero" | "mid"
) -> jnp.ndarray:

    names = list(robot.joints.actuated_names)
    lowers = np.array(robot.joints.lower_limits)
    uppers = np.array(robot.joints.upper_limits)
    assert len(names) == lowers.shape[0] == uppers.shape[0] == robot.joints.num_actuated_joints

    vals = []
    for name, lo, hi in zip(names, lowers, uppers):
        if name in uniform_ranges:
            a, b = uniform_ranges[name]
            v = float(rng.uniform(a, b))
        elif name in fixed_values:
            v = float(fixed_values[name])
        else:
            if default_mode == "mid":
                v = 0.5 * (float(lo) + float(hi))
            else:
                v = 0.0
        v = float(np.clip(v, lo, hi))
        vals.append(v)

    return jnp.array(vals, dtype=jnp.float32)


def build_pose_aa(dof_seq: np.ndarray, root_quat_seq_wxyz: np.ndarray) -> np.ndarray:
    # Use the fixed G1 axis mapping from the reference script (23 joints)
    g1_rotation_axis = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    num_frames = dof_seq.shape[0]
    pose_aa = np.zeros((num_frames, 1 + 29, 3), dtype=np.float32)
    for i in range(num_frames):
        pose_aa[i, 0] = wxyz_to_axis_angle(root_quat_seq_wxyz[i])
        for j in range(min(29, dof_seq.shape[1])):
            pose_aa[i, j + 1] = g1_rotation_axis[j] * dof_seq[i, j]
    return pose_aa



def main(config: AppConfig):

    logger = logging.getLogger("g1_sampler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(config.sample.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    motions = {}
    num_generated = 0
    sequence_index = 0

    output_path = Path(config.sample.output_path)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = output_path / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.sample.num_samples, desc="Generating sequences")

    urdf = yourdfpy.URDF.load(config.sample.urdf_path)

    while num_generated < config.sample.num_samples:

        
        robot = pk.Robot.from_urdf(urdf)
        actuated_names = [robot.joints.names[i] for i in config.sample.actuated_index]  
            
        # initial_pos_list = []
        # for joint_name, lower, upper in zip(
        #     actuated_names, robot.joints.lower_limits, robot.joints.upper_limits
        # ):
        #     if joint_name in config.sample.default_joints.keys():
        #         initial_pos = config.sample.default_joints[joint_name]
        #     else:
        #         initial_pos = 0.0

        #     # print(f"name: {joint_name}, initial_pos: {initial_pos}, lower: {lower}, upper: {upper}")
        #     assert (
        #         initial_pos < upper and initial_pos > lower
        #     ), f"initial_pos {initial_pos} is not within joint limits {lower} and {upper}"
        #     initial_pos = np.clip(initial_pos, lower, upper)
        #     initial_pos_list.append(initial_pos)

        # print("initial_pos_list", initial_pos_list)

        # initial_pos_list = jnp.array(initial_pos_list)
        # robot = pk.Robot.from_urdf(urdf, default_joint_cfg=initial_pos_list)

        rng = np.random.default_rng(config.sample.seed + sequence_index)
        urdf = yourdfpy.URDF.load(config.sample.urdf_path)
        robot_tmp = pk.Robot.from_urdf(urdf)

        if config.sample.randomize_init:
            init_cfg = sample_init_cfg_actuated_order(
				robot=robot_tmp,
				fixed_values=config.sample.default_joints,
				uniform_ranges=config.sample.random_init_ranges,
				rng=rng,
				default_mode="zero",
			)
        else:
            init_cfg = sample_init_cfg_actuated_order(
				robot=robot_tmp,
				fixed_values=config.sample.default_joints,
				uniform_ranges={},
				rng=rng,
				default_mode="zero",
			)

		# Now create the actual robot used in optimization
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=init_cfg)

        # from pyroki.collision import RobotCollision
        # robot_coll = RobotCollision.from_urdf(urdf, ignore_immediate_adjacents=True)
        sides = (
            ["left", "right"] if config.sample.side == "both" else [config.sample.side]
        )
        for support_side in sides:
            if num_generated >= config.sample.num_samples:
                break
            support_idx = 0 if support_side == "left" else 1

            tries = 0
            while True:
                # Sample floating foot pose
                pos_f, wxyz_f = sample_floating_foot_pose(
                    support_is_left=(support_idx == 0),
                    x_range=tuple(config.sample.float_x_range),
                    y_abs_range=tuple(config.sample.float_y_abs_range),
                    z_range=tuple(config.sample.float_z_range),
                )
                print(config.sample.float_z_range)
                pelvis_h = float(
                    np.random.uniform(
                        config.sample.pelvis_height_range[0],
                        config.sample.pelvis_height_range[1],
                    )
                )

                # overwrite the target foot pose
                # pos_f = np.array([0.64, -0.25 if support_idx == 0.25 else 0.00, 0.3], dtype=np.float32)
                # wxyz_f = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                # pelvis_h = 0.35

                print(f"target foot is {'left' if support_idx == 0 else 'right'}")
                print(f"pelvis height is {pelvis_h}")
                print(f"target foot pos is {pos_f}")
                print(f"target foot wxyz is {wxyz_f}")

                # Optimize sequence
                print(init_cfg)

                from pyroki.collision import HalfSpace

                plane_coll = HalfSpace.from_point_and_normal(
                    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
                )
                world_coll_list = []
                world_coll_list.append(plane_coll)
                qpos_seq, had_nans = optimize_motion_sequence(
                    robot=robot,
                    urdf=urdf,
                    support_foot_index=support_idx,
                    target_foot_pose=(pos_f, wxyz_f),
                    pelvis_height=pelvis_h,
                    config=config.seq,
                    weights=config.weights,
                    geometry=config.geom,
                    opt_params=config.opt,
                    init_cfg=init_cfg,
                    world_coll_list=world_coll_list,
                    actuated_index=config.sample.actuated_index,
                    upper_body_joints=config.sample.upper_body_joints,
                )

                if had_nans:
                    logger.info(
                        f"DISCARD NaNs: side={support_side} pelvis={pelvis_h:.3f} pos={pos_f.tolist()} wxyz={wxyz_f.tolist()}"
                    )
                    tries += 1
                    try:
                        jax.clear_caches()
                    except Exception:
                        pass
                    if tries >= config.sample.max_retries:
                        logger.info(
                            f"GIVE UP after {tries} retries for side={support_side}; moving on"
                        )
                        break
                    continue

                # Success
                logger.info(
                    f"SUCCESS: side={support_side} pelvis={pelvis_h:.3f} steps={qpos_seq.shape[0]} pos={pos_f.tolist()}"
                )


                is_stable = validate_motion_stability(
                    robot=robot,
                    urdf=urdf,
                    qpos_seq=qpos_seq,
                    support_foot_index=support_idx,
                    geometry=config.geom,
                    opt_params=config.opt,
                )
                # is_stable = True
                if not is_stable:
                    logger.info(
                        f"DISCARD Unstable: side={support_side} pelvis={pelvis_h:.3f} pos={pos_f.tolist()} wxyz={wxyz_f.tolist()}"
                    )
                    tries += 1
                    if tries >= config.sample.max_retries:
                        logger.info(
                            f"GIVE UP after {tries} retries for side={support_side};"
                        )
                        break
                    continue

                logger.info(f"Stability validation PASSED for side={support_side}")
                break

            # Append standstill frames
            if config.sample.stand_frames > 0:
                last = qpos_seq[-1]
                stand_seq = np.repeat(last[None, :], config.sample.stand_frames, axis=0)
                qpos_seq = np.concatenate([qpos_seq, stand_seq], axis=0)

            # Create symmetric motion: init -> target -> init
            # Reverse the trajectory (excluding standstill frames if any)
            if config.sample.stand_frames > 0:
                # If we have standstill frames, reverse only the motion part
                motion_part = qpos_seq[: -config.sample.stand_frames]
                reverse_motion = motion_part[::-1]  # Reverse the motion sequence
                # Concatenate: init->target + standstill + target->init
                qpos_seq = np.concatenate([qpos_seq, reverse_motion], axis=0)
            else:
                # No standstill frames, just reverse the entire sequence
                reverse_seq = qpos_seq[::-1]  # Reverse the sequence
                # Concatenate: init->target + target->init
                qpos_seq = np.concatenate([qpos_seq, reverse_seq], axis=0)


            root_pos_seq = qpos_seq[:, 0:3].astype(np.float32)
            root_quat_wxyz = qpos_seq[:, 3:7].astype(np.float64)
            root_quat_xyzw = np.stack([wxyz_to_xyzw(q) for q in root_quat_wxyz], axis=0)
            cfg = qpos_seq[:, 7:]
            dof = cfg
            pose_aa = build_pose_aa(dof, root_quat_wxyz)

            motion_entry = {
                "root_trans_offset": root_pos_seq,
                "pose_aa": pose_aa,
                "dof": dof,
                "root_rot": root_quat_xyzw,
                "fps": int(config.sample.fps),
                "stance_leg": ("left" if support_idx == 0 else "right"),
            }
            
            key = f"single_stand_pose_{'left' if support_idx == 0 else 'right'}_{sequence_index:03d}"
            motions[key] = motion_entry
            
            out_file = out_dir / f"{key}.pkl"
            joblib.dump({key: motion_entry}, out_file)

            num_generated += 1
            sequence_index += 1
            try:
                pbar.set_postfix({"side": support_side, "idx": sequence_index - 1})
            except Exception:
                pass
            pbar.update(1)

            try:
                jax.clear_caches()
            except Exception:
                pass

    pbar.close()


    print(f"Saved {len(motions)} motions to {out_dir}")


if __name__ == "__main__":
    config = tyro.cli(AppConfig)
    main(config)

# python examples/g1_sample_sequences.py   --num_samples 6  --side both   --num_frames 45 --stand_frames 60   --support_foot_height 0.03      --output motions/g1/sampled_static_poses_v4.pkl  (e.g. 6 samples)
