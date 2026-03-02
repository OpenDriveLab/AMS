#!/usr/bin/env python3
"""
Auto check robot motions for abnormalities:
1. Self-collision: Collisions between robot body parts
2. Ground penetration: Body parts penetrating below ground level

Judgment criteria:
- Self-collision: Collisions between robot body parts exceeding thresholds
- Ground penetration: Body parts below threshold height (default: -1cm) for > 0.5s OR > 5% of frames
"""

import argparse
import json
import os
import gc
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import joblib
import numpy as np
import mujoco as mj
import pathlib
import psutil



DEFAULT_FOOT_KEYS = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_toe_link",
    "right_toe_link",
    "left_foot_link",
    "right_foot_link",
]

SELF_COLLISION_MIN_SECONDS = 0.6
SELF_COLLISION_MIN_RATIO = 0.10

# Ground penetration detection thresholds
GROUND_PENETRATION_THRESHOLD = -0.08 # Body parts below -3cm are considered penetrating ground
GROUND_PENETRATION_MIN_SECONDS = 0.5  # Minimum continuous seconds of penetration to flag
GROUND_PENETRATION_MIN_RATIO = 0.03 # Minimum ratio of frames with penetration to flag (5%)

# contact_dict-style filters to keep only meaningful self-collisions
# `pairs`: optional whitelist of (body_a, body_b) tuples (sorted names)
# `min_penetration`: minimum penetration depth (in meters) to count a collision
SELF_COLLISION_CONTACT_DICT = {
    "default": {
        "pairs": [],
        "min_penetration": 0.02,
    },
}


HERE = pathlib.Path(__file__).parent
ASSET_ROOT = HERE / ".." / "robot"
ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_description" / "mjcf" / "g1_sysid_29dof.xml",
}

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = joblib.load(f)
        if isinstance(motion_data, dict) and len(motion_data) == 1:
            inner_key = list(motion_data.keys())[0]
            motion_data = motion_data[inner_key]
        motion_fps = motion_data["fps"]
        if motion_data.get("root_pos") is not None:
            motion_root_pos = motion_data["root_pos"]
        else:
            motion_root_pos = motion_data["root_trans_offset"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        if motion_data.get("dof_pos") is not None:
            motion_dof_pos = motion_data["dof_pos"]
        else:
            motion_dof_pos = motion_data["dof"]
        if motion_data.get("local_body_pos") is not None:
            motion_local_body_pos = motion_data["local_body_pos"]
        else:
            motion_local_body_pos = None
        if motion_data.get("link_body_list") is not None:
            motion_link_body_list = motion_data["link_body_list"]
        else:
            motion_link_body_list = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'left_toe_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'right_toe_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link', 'head_mocap', 'imu_in_torso', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list





@dataclass
class CheckReport:
    motion_file: str
    is_abnormal: bool  # True if self-collision or ground penetration
    self_collision: "SelfCollisionInfo"
    ground_penetration: "GroundPenetrationInfo"


@dataclass
class SelfCollisionInfo:
    has_self_collision: bool
    total_frames: int
    collision_frames: int
    collision_ratio: float
    max_continuous_collision_frames: int
    max_continuous_collision_seconds: float
    max_continuous_collision_ratio: float
    total_collision_events: int
    top_collision_pairs: List[str]
    # Consecutive collision segments (1-based inclusive)
    collision_segments: List[Dict[str, int]] = field(default_factory=list)
    violation_reason: Optional[str] = None


@dataclass
class GroundPenetrationInfo:
    has_penetration: bool
    total_frames: int
    fps: float
    penetration_frames: int
    penetration_ratio: float
    max_penetration_depth: float  # Maximum depth below ground (meters, negative value)
    max_penetration_frame: int
    max_penetration_body: str
    max_continuous_penetration_frames: int
    max_continuous_penetration_seconds: float
    max_continuous_penetration_ratio: float
    penetrating_bodies: List[str] = field(default_factory=list)  # Bodies that penetrate ground
    # Consecutive penetration segments (1-based inclusive)
    penetration_segments: List[Dict[str, int]] = field(default_factory=list)
    violation_reason: Optional[str] = None


def get_mujoco_rendered_data(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    robot_type: str,
) -> tuple:
    if robot_type not in ROBOT_XML_DICT:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    xml_file = ROBOT_XML_DICT[robot_type]
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    
    total_frames = len(root_pos)
    num_dof = len(dof_pos[0]) if len(dof_pos.shape) > 1 else 0
    num_bodies = model.nbody
    num_geoms = model.ngeom
    
    actual_dof_pos = np.zeros((total_frames, num_dof))
    actual_body_pos = np.zeros((total_frames, num_bodies, 3))
    body_names = []
    geom_names = []
    collisions_by_frame: List[List[Tuple[str, str, float]]] = [[] for _ in range(total_frames)]
    ground_contacts_by_frame: List[List[Tuple[str, float]]] = [[] for _ in range(total_frames)]  # (geom_name, penetration_depth)
    
    filter_cfg = SELF_COLLISION_CONTACT_DICT.get(
        robot_type, SELF_COLLISION_CONTACT_DICT.get("default", {})
    )
    pair_whitelist = {
        tuple(sorted(pair)) for pair in filter_cfg.get("pairs", []) if len(pair) == 2
    }
    min_penetration = filter_cfg.get("min_penetration", 0.0)
    
    # Get body and geom names
    for i in range(num_bodies):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        body_names.append(name if name is not None else f"body_{i}")
    for i in range(num_geoms):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        geom_names.append(name if name is not None else f"geom_{i}")
    
    # Render each frame through MuJoCo
    for frame_idx in range(total_frames):
        # Set joint positions
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]  # quat scalar first for MuJoCo
        data.qpos[7:] = dof_pos[frame_idx]
        
        # Forward kinematics + collision detection
        mj.mj_forward(model, data)
        
        # Get actual joint angles (after potential clipping)
        actual_dof_pos[frame_idx] = data.qpos[7:].copy()
        
        # Get actual body positions in world frame
        actual_body_pos[frame_idx] = data.xpos.copy()
        
        # Collect self-collisions and ground contacts
        if data.ncon > 0:
            frame_collisions = collisions_by_frame[frame_idx]
            frame_ground_contacts = ground_contacts_by_frame[frame_idx]
            for c_idx in range(data.ncon):
                contact = data.contact[c_idx]
                geom1 = contact.geom1
                geom2 = contact.geom2
                if geom1 < 0 or geom2 < 0:
                    continue
                body1 = model.geom_bodyid[geom1]
                body2 = model.geom_bodyid[geom2]
                
                # Check for ground contact (one body is world/ground, body id 0)
                is_ground_contact = (body1 == 0) or (body2 == 0)
                if is_ground_contact:
                    # This is a contact with ground
                    robot_body = body1 if body2 == 0 else body2
                    robot_geom = geom1 if body2 == 0 else geom2
                    if robot_body > 0:  # Valid robot body
                        geom_name = geom_names[robot_geom] if robot_geom < len(geom_names) else f"geom_{robot_geom}"
                        body_name = body_names[robot_body] if robot_body < len(body_names) else f"body_{robot_body}"
                        # Penetration depth: negative contact.dist means penetration
                        penetration = max(0.0, -float(contact.dist))
                        if penetration > 0:  # Only record actual penetration (not just contact)
                            frame_ground_contacts.append((body_name, geom_name, penetration))
                    continue
                
                # Self-collisions (exclude world/environment geoms)
                if body1 <= 0 or body2 <= 0:
                    continue
                body_pair = tuple(sorted((body_names[body1], body_names[body2])))
                if body1 == body2:
                    continue
                # Only consider contacts with actual penetration (dist < 0)
                # contact.dist: positive = gap, negative = penetration
                if contact.dist >= 0:
                    continue  # Skip contacts with gap (no penetration)
                
                if pair_whitelist and body_pair not in pair_whitelist:
                    continue
                penetration = -float(contact.dist)  # penetration depth (always positive when dist < 0)
                if penetration < min_penetration:
                    continue
                frame_collisions.append(
                    (body_pair[0], body_pair[1], penetration)
                )
    
    return actual_dof_pos, actual_body_pos, body_names, collisions_by_frame, ground_contacts_by_frame


def find_max_continuous_off_ground(mask: np.ndarray) -> tuple:
    """Find maximum continuous sequence of True values in mask."""
    if mask.size == 0:
        return (0, -1, -1)
    
    max_length = 0
    max_start = -1
    max_end = -1
    
    current_length = 0
    current_start = -1
    
    for i, value in enumerate(mask):
        if value:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i
        else:
            current_length = 0
            current_start = -1
    
    return (max_length, max_start, max_end)


def mask_to_segments(mask: np.ndarray) -> List[Dict[str, int]]:
    """Convert boolean mask to list of 1-based inclusive segments."""
    segs: List[Dict[str, int]] = []
    if mask.size == 0:
        return segs
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            end = i - 1
            segs.append(
                {
                    "start_frame": start + 1,
                    "end_frame": end + 1,
                    "num_frames": end - start + 1,
                }
            )
            start = None
    if start is not None:
        end = len(mask) - 1
        segs.append(
            {
                "start_frame": start + 1,
                "end_frame": end + 1,
                "num_frames": end - start + 1,
            }
        )
    return segs


def detect_self_collisions(
    collisions_by_frame: List[List[Tuple[str, str, float]]],
    fps: float,
    min_seconds: float = SELF_COLLISION_MIN_SECONDS,
    min_ratio: float = SELF_COLLISION_MIN_RATIO,
) -> SelfCollisionInfo:
    total_frames = len(collisions_by_frame)
    if total_frames == 0:
        return SelfCollisionInfo(
            has_self_collision=False,
            total_frames=0,
            collision_frames=0,
            collision_ratio=0.0,
            max_continuous_collision_frames=0,
            max_continuous_collision_seconds=0.0,
            max_continuous_collision_ratio=0.0,
            total_collision_events=0,
            top_collision_pairs=[],
            violation_reason=None,
        )
    
    collision_mask = np.array([len(frame) > 0 for frame in collisions_by_frame], dtype=bool)
    collision_frames = int(collision_mask.sum())
    collision_ratio = collision_frames / total_frames if total_frames > 0 else 0.0
    
    max_frames, start_idx, end_idx = find_max_continuous_off_ground(collision_mask)
    max_seconds = max_frames / fps if fps > 0 else 0.0
    max_ratio = max_frames / total_frames if total_frames > 0 else 0.0
    
    # Count collision pairs and record frame indices
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_frames: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    total_events = 0
    for frame_idx, frame in enumerate(collisions_by_frame):
        for geom1, geom2, _ in frame:
            key = tuple(sorted((geom1, geom2)))
            pair_counts[key] += 1
            pair_frames[key].append(frame_idx)
            total_events += 1
    
    # Format top pairs with frame information (exact frame list preview)
    top_pairs = []
    for (a, b), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        frames = sorted(set(pair_frames[(a, b)]))
        n_unique = len(frames)
        if n_unique == 0:
            frame_str = "frames=[]"
        elif n_unique <= 20:
            frame_str = f"frames={frames}"
        else:
            head = frames[:10]
            tail = frames[-10:]
            frame_str = f"frames={head} ... {tail} (unique_frames={n_unique})"
        top_pairs.append(f"{a} <-> {b} ({count} contacts, {frame_str})")
    
    has_self_collision = (collision_ratio >= min_ratio) or (max_seconds >= min_seconds)
    violation_reason = None
    if has_self_collision:
        reason_parts = []
        if collision_ratio >= min_ratio:
            reason_parts.append(f"collision ratio {collision_ratio:.2%} ≥ {min_ratio:.2%}")
        if max_seconds >= min_seconds:
            reason_parts.append(f"continuous {max_seconds:.2f}s ≥ {min_seconds:.2f}s")
        violation_reason = "; ".join(reason_parts)
    
    return SelfCollisionInfo(
        has_self_collision=has_self_collision,
        total_frames=total_frames,
        collision_frames=collision_frames,
        collision_ratio=collision_ratio,
        max_continuous_collision_frames=max_frames,
        max_continuous_collision_seconds=max_seconds,
        max_continuous_collision_ratio=max_ratio,
        total_collision_events=total_events,
        top_collision_pairs=top_pairs,
        collision_segments=mask_to_segments(collision_mask),
        violation_reason=violation_reason,
    )


def detect_ground_penetration(
    ground_contacts_by_frame: List[List[Tuple[str, str, float]]],
    fps: float,
    penetration_threshold: float = GROUND_PENETRATION_THRESHOLD,
    min_seconds: float = GROUND_PENETRATION_MIN_SECONDS,
    min_ratio: float = GROUND_PENETRATION_MIN_RATIO,
    foot_names: List[str] = None,
) -> GroundPenetrationInfo:
    """
    Detect if any body parts are penetrating the ground using geom-ground collision data.
    
    This is more accurate than checking body positions because:
    1. Geoms are the actual geometric shapes (arms, legs, etc.)
    2. MuJoCo collision detection accurately calculates penetration depth
    3. More precise than checking body center of mass positions
    
    Args:
        ground_contacts_by_frame: List of ground contacts per frame, each contact is (body_name, geom_name, penetration_depth)
        fps: Frames per second
        penetration_threshold: Minimum penetration depth to consider significant (meters, default: -0.01m)
        min_seconds: Minimum continuous seconds of penetration to flag
        min_ratio: Minimum ratio of frames with penetration to flag
        foot_names: List of foot body names to exclude from penetration detection (default: DEFAULT_FOOT_KEYS)
    
    Returns:
        GroundPenetrationInfo with detection results
    """
    if foot_names is None:
        foot_names = DEFAULT_FOOT_KEYS
    
    # Create a set of foot-related keywords for faster lookup
    foot_keywords = set()
    for foot_name in foot_names:
        foot_keywords.add(foot_name.lower())
        # Also add common foot-related keywords
        if "ankle" in foot_name.lower():
            foot_keywords.add("ankle")
        if "foot" in foot_name.lower():
            foot_keywords.add("foot")
        if "toe" in foot_name.lower():
            foot_keywords.add("toe")
    
    def is_foot_body(body_name: str) -> bool:
        """Check if a body name is related to feet."""
        body_lower = body_name.lower()
        # Check if body name contains any foot-related keywords
        return any(keyword in body_lower for keyword in foot_keywords)
    
    total_frames = len(ground_contacts_by_frame)
    if total_frames == 0:
        return GroundPenetrationInfo(
            has_penetration=False,
            total_frames=total_frames,
            fps=fps,
            penetration_frames=0,
            penetration_ratio=0.0,
            max_penetration_depth=0.0,
            max_penetration_frame=-1,
            max_penetration_body="",
            max_continuous_penetration_frames=0,
            max_continuous_penetration_seconds=0.0,
            max_continuous_penetration_ratio=0.0,
            penetrating_bodies=[],
            violation_reason=None,
        )
    
    # Check all frames for ground penetration using collision data
    penetration_mask = np.zeros(total_frames, dtype=bool)  # True if any geom penetrates in this frame
    penetration_depths = np.zeros(total_frames)  # Maximum penetration depth per frame
    penetration_bodies_per_frame: List[List[str]] = [[] for _ in range(total_frames)]
    
    # Track which bodies penetrate most frequently
    body_penetration_counts: Dict[str, int] = defaultdict(int)
    
    for frame_idx in range(total_frames):
        frame_contacts = ground_contacts_by_frame[frame_idx]
        frame_penetration_depth = 0.0
        frame_penetrating_bodies = []
        
        for body_name, geom_name, penetration_depth in frame_contacts:
            # Skip foot-ground contacts (feet are supposed to contact ground)
            if is_foot_body(body_name):
                continue
            
            # Only consider penetrations above threshold (penetration_depth is already positive)
            if penetration_depth > abs(penetration_threshold):
                if penetration_depth > frame_penetration_depth:
                    frame_penetration_depth = penetration_depth
                if body_name not in frame_penetrating_bodies:
                    frame_penetrating_bodies.append(body_name)
                body_penetration_counts[body_name] += 1
        
        if frame_penetration_depth > 0:
            penetration_mask[frame_idx] = True
            penetration_depths[frame_idx] = frame_penetration_depth
            penetration_bodies_per_frame[frame_idx] = frame_penetrating_bodies
    
    penetration_frames = int(penetration_mask.sum())
    penetration_ratio = penetration_frames / total_frames if total_frames > 0 else 0.0
    
    # Find maximum penetration depth and frame
    max_penetration_depth = float(np.max(penetration_depths)) if penetration_frames > 0 else 0.0
    max_penetration_frame = int(np.argmax(penetration_depths)) if penetration_frames > 0 else -1
    max_penetration_body = (
        penetration_bodies_per_frame[max_penetration_frame][0]
        if max_penetration_frame >= 0 and penetration_bodies_per_frame[max_penetration_frame]
        else ""
    )
    
    # Find maximum continuous penetration
    max_frames, start_idx, end_idx = find_max_continuous_off_ground(penetration_mask)
    max_seconds = max_frames / fps if fps > 0 else 0.0
    max_ratio = max_frames / total_frames if total_frames > 0 else 0.0
    
    # Get bodies that penetrate most frequently (top 5)
    top_penetrating_bodies = [
        body_name for body_name, _ in sorted(body_penetration_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    
    # Determine if penetration is significant
    has_penetration = (penetration_ratio >= min_ratio) or (max_seconds >= min_seconds)
    violation_reason = None
    if has_penetration:
        reason_parts = []
        if penetration_ratio >= min_ratio:
            reason_parts.append(f"penetration ratio {penetration_ratio:.2%} ≥ {min_ratio:.2%}")
        if max_seconds >= min_seconds:
            reason_parts.append(f"continuous {max_seconds:.2f}s ≥ {min_seconds:.2f}s")
        violation_reason = "; ".join(reason_parts)
    
    return GroundPenetrationInfo(
        has_penetration=has_penetration,
        total_frames=total_frames,
        fps=fps,
        penetration_frames=penetration_frames,
        penetration_ratio=penetration_ratio,
        max_penetration_depth=max_penetration_depth,
        max_penetration_frame=max_penetration_frame,
        max_penetration_body=max_penetration_body,
        max_continuous_penetration_frames=max_frames,
        max_continuous_penetration_seconds=max_seconds,
        max_continuous_penetration_ratio=max_ratio,
        penetrating_bodies=top_penetrating_bodies,
        penetration_segments=mask_to_segments(penetration_mask),
        violation_reason=violation_reason,
    )


def check_motion(
    motion_file: str,
    robot_type: str,
    foot_names: List[str] = None,
) -> Optional[CheckReport]:
    """Check a single motion file for self-collision and ground penetration."""
    if foot_names is None:
        foot_names = DEFAULT_FOOT_KEYS
    
    # Load motion data
    try:
        (
            motion_data,
            fps,
            root_pos,
            root_rot,
            dof_pos,
            local_body_pos,
            link_body_list,
        ) = load_robot_motion(motion_file)
    except Exception as e:
        print(f"[WARN] Failed to load {motion_file}: {e}")
        return None
    
    # Render through MuJoCo to get actual data (joint angles may be clipped, body positions from MuJoCo)
    try:
        (
            actual_dof_pos,
            actual_body_pos,
            mujoco_body_names,
            collisions_by_frame,
            ground_contacts_by_frame,
        ) = get_mujoco_rendered_data(root_pos, root_rot, dof_pos, robot_type)
    except Exception as e:
        print(f"[WARN] Failed to render through MuJoCo for {motion_file}: {e}")
        # Fallback to original data
        actual_dof_pos = dof_pos
        mujoco_body_names = link_body_list
        collisions_by_frame = [[] for _ in range(len(dof_pos))]
        ground_contacts_by_frame = [[] for _ in range(len(dof_pos))]
    
    # Detect self-collisions using collision data
    self_collision_info = detect_self_collisions(
        collisions_by_frame,
        fps,
        min_seconds=SELF_COLLISION_MIN_SECONDS,
        min_ratio=SELF_COLLISION_MIN_RATIO,
    )
    
    # Detect ground penetration using geom-ground collision data (more accurate than body positions)
    # Exclude foot-ground contacts (feet are supposed to contact ground)
    ground_penetration_info = detect_ground_penetration(
        ground_contacts_by_frame,  # Use geom-ground collision data from MuJoCo
        fps,
        penetration_threshold=GROUND_PENETRATION_THRESHOLD,
        min_seconds=GROUND_PENETRATION_MIN_SECONDS,
        min_ratio=GROUND_PENETRATION_MIN_RATIO,
        foot_names=foot_names,  # Exclude foot bodies from penetration detection
    )
    
    # Determine if motion is abnormal (only check self-collision and ground penetration)
    is_abnormal = (
        self_collision_info.has_self_collision
        or ground_penetration_info.has_penetration
    )
    
    # Use relative path for motion_file
    return CheckReport(
        motion_file=motion_file,
        is_abnormal=is_abnormal,
        self_collision=self_collision_info,
        ground_penetration=ground_penetration_info,
    )


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)
    return 0.0


def check_memory_threshold(warning_threshold_mb: float = 8000, critical_threshold_mb: float = 12000) -> bool:
    """Check if memory usage exceeds thresholds. Returns True if critical."""
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 ** 2)
    
    if memory_mb > critical_threshold_mb:
        print(f"[CRITICAL] Memory usage: {memory_mb:.1f} MB exceeds critical threshold ({critical_threshold_mb:.1f} MB)")
        return True
    elif memory_mb > warning_threshold_mb:
        print(f"[WARNING] Memory usage: {memory_mb:.1f} MB exceeds warning threshold ({warning_threshold_mb:.1f} MB)")
    
    return False


def _save_progress(progress_path: Path, processed_files: set, reports: List[CheckReport]):
    """Save progress to JSON file."""
    try:
        progress_data = {
            "processed_files": list(processed_files),
            "reports": [asdict(r) for r in reports],
            "timestamp": datetime.now().isoformat(),
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save progress: {e}")


def scan_folder(
    motion_folder: str,
    robot_type: str,
    foot_names: List[str] = None,
    batch_size: int = 100,
    save_progress: bool = True,
    progress_file: str = None,
    resume_from_progress: bool = True,
) -> List[CheckReport]:
    """Scan folder for abnormal motions with memory-efficient batch processing.
    
    Args:
        motion_folder: Folder containing motion files
        robot_type: Robot type
        batch_size: Number of files to process before clearing memory (default: 100)
        save_progress: Whether to save progress periodically (default: True)
        progress_file: Path to save progress JSON (default: auto_check_progress.json)
        resume_from_progress: Whether to resume from previous progress file (default: True). 
                              If False, start from scratch and overwrite existing progress.
    """
    motion_folder = Path(motion_folder)
    if not motion_folder.exists():
        raise FileNotFoundError(f"Motion folder does not exist: {motion_folder}")
    
    reports = []
    motion_files = []
    
    # Collect all pkl files
    for root, _, files in os.walk(motion_folder):
        for file in files:
            if file.endswith(".pkl"):
                motion_files.append(Path(root) / file)
    
    total_files = len(motion_files)
    print(f"Found {total_files} motion files, scanning...")
    
    initial_memory = get_memory_usage_mb()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Load existing progress if available (only if resume_from_progress is True)
    if progress_file is None:
        progress_file = "auto_check_progress.json"
    progress_path = Path(progress_file)
    processed_files = set()
    
    if progress_path.exists() and save_progress and resume_from_progress:
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get("processed_files", []))
                # Load existing reports
                existing_reports = progress_data.get("reports", [])
                reports = []
                for r in existing_reports:
                    try:
                        # Convert dict to CheckReport, handling missing fields
                        if isinstance(r, dict):
                            # Ensure all required fields are present
                            # Convert nested dicts to dataclass instances
                            if "self_collision" in r and isinstance(r["self_collision"], dict):
                                r["self_collision"] = SelfCollisionInfo(**r["self_collision"])
                            if "ground_penetration" in r and isinstance(r["ground_penetration"], dict):
                                r["ground_penetration"] = GroundPenetrationInfo(**r["ground_penetration"])
                            elif "ground_penetration" not in r:
                                # Create default GroundPenetrationInfo for old progress files
                                total_frames = 0
                                fps = 30.0
                                if "self_collision" in r and isinstance(r["self_collision"], dict):
                                    total_frames = r["self_collision"].get("total_frames", 0)
                                r["ground_penetration"] = GroundPenetrationInfo(
                                    has_penetration=False,
                                    total_frames=total_frames,
                                    fps=fps,
                                    penetration_frames=0,
                                    penetration_ratio=0.0,
                                    max_penetration_depth=0.0,
                                    max_penetration_frame=-1,
                                    max_penetration_body="",
                                    max_continuous_penetration_frames=0,
                                    max_continuous_penetration_seconds=0.0,
                                    max_continuous_penetration_ratio=0.0,
                                    penetrating_bodies=[],
                                    violation_reason=None,
                                )
                            reports.append(CheckReport(**r))
                        elif isinstance(r, CheckReport):
                            reports.append(r)
                    except Exception as e:
                        print(f"[WARN] Failed to load report from progress file: {e}")
                        continue
                print(f"Loaded {len(processed_files)} processed files from progress file")
        except Exception as e:
            print(f"[WARN] Failed to load progress file: {e}")
    
    # Filter out already processed files
    remaining_files = [f for f in motion_files if str(f.relative_to(motion_folder)) not in processed_files]
    print(f"Remaining files to process: {len(remaining_files)}")
    
    # Process files in batches
    for batch_idx in range(0, len(remaining_files), batch_size):
        batch_files = remaining_files[batch_idx:batch_idx + batch_size]
        batch_start = batch_idx + 1
        batch_end = min(batch_idx + batch_size, len(remaining_files))
        
        print(f"\n[Batch {batch_idx // batch_size + 1}] Processing files {batch_start}-{batch_end} of {len(remaining_files)}")
        
        batch_reports = []
        for file_idx, motion_file in enumerate(batch_files):
            rel_path = str(motion_file.relative_to(motion_folder))
            
            try:
                report = check_motion(
                    str(motion_file),
                    robot_type,
                    foot_names=foot_names,
                )
                if report:
                    report.motion_file = rel_path
                    batch_reports.append(report)
                    reports.append(report)
                    processed_files.add(rel_path)
            except MemoryError as e:
                print(f"\n[ERROR] Memory error processing {rel_path}: {e}")
                print("[ERROR] Try reducing batch_size or processing fewer files at once")
                # Save progress before exiting
                if save_progress:
                    _save_progress(progress_path, processed_files, reports)
                raise
            except Exception as e:
                print(f"[WARN] Failed to process {rel_path}: {e}")
                # Still mark as processed to avoid retrying
                processed_files.add(rel_path)
            
            # Print progress every 10 files
            if (file_idx + 1) % 10 == 0:
                current_memory = get_memory_usage_mb()
                print(f"  Processed {file_idx + 1}/{len(batch_files)} files"
                      + (f", Memory: {current_memory:.1f} MB"))
        
        # Memory cleanup after each batch
        current_memory = get_memory_usage_mb()
        print(f"Batch complete. Memory usage: {current_memory:.1f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Check memory threshold
        if check_memory_threshold():
            print("[WARNING] High memory usage detected. Consider reducing batch_size.")
        
        # Save progress after each batch
        if save_progress:
            _save_progress(progress_path, processed_files, reports)
            print(f"Progress saved: {len(processed_files)}/{total_files} files processed")
    
    # Final cleanup
    gc.collect()
    
    final_memory = get_memory_usage_mb()
    print(f"\nFinal memory usage: {final_memory:.1f} MB")
    if initial_memory > 0:
        print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Auto check robot motions for bad collisions"
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        required=True,
        help="Folder containing robot *.pkl files",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        help=f"Robot type. Available: {list(ROBOT_XML_DICT.keys())}",
    )
    parser.add_argument(
        "--foot_names",
        type=str,
        nargs="*",
        default=None,
        help=f"Foot link names to check (default: {DEFAULT_FOOT_KEYS})",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save check reports as JSON (default: auto_check_reports.json)",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Check single motion file instead of folder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of files to process before clearing memory (default: 100). Reduce if memory issues occur.",
    )
    parser.add_argument(
        "--no_save_progress",
        action="store_true",
        help="Disable periodic progress saving (default: enabled)",
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default=None,
        help="Path to save/load progress JSON (default: auto_check_progress.json)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from previous progress file, start from scratch (overwrite mode)",
    )
    
    args = parser.parse_args()
    
    foot_names = args.foot_names if args.foot_names else DEFAULT_FOOT_KEYS
    
    if args.motion_file:
        # Single file mode - only output if abnormal
        report = check_motion(
            args.motion_file,
            args.robot_type,
            foot_names=foot_names,
        )
        if report:
            if report.is_abnormal:
                print(f"\n[ABNORMAL] Motion: {report.motion_file}")
            if report.self_collision.has_self_collision:
                print(f"  Self-collision: {report.self_collision.has_self_collision}")
                print(f"    Collision ratio: {report.self_collision.collision_ratio:.2%}")
                print(f"    Max continuous: {report.self_collision.max_continuous_collision_seconds:.2f}s "
                      f"({report.self_collision.max_continuous_collision_ratio:.2%})")
                if report.self_collision.top_collision_pairs:
                    print("    Top pairs:")
                    for pair in report.self_collision.top_collision_pairs:
                        print(f"      - {pair}")
                if report.self_collision.violation_reason:
                    print(f"    Reason: {report.self_collision.violation_reason}")
            if report.ground_penetration.has_penetration:
                print(f"  Ground penetration: {report.ground_penetration.has_penetration}")
                print(f"    Penetration ratio: {report.ground_penetration.penetration_ratio:.2%}")
                print(f"    Max penetration depth: {report.ground_penetration.max_penetration_depth:.4f} m "
                      f"at frame {report.ground_penetration.max_penetration_frame} (body: {report.ground_penetration.max_penetration_body})")
                print(f"    Max continuous: {report.ground_penetration.max_continuous_penetration_seconds:.2f}s "
                      f"({report.ground_penetration.max_continuous_penetration_ratio:.2%})")
                if report.ground_penetration.penetrating_bodies:
                    print("    Top penetrating bodies:")
                    for body in report.ground_penetration.penetrating_bodies:
                        print(f"      - {body}")
                if report.ground_penetration.violation_reason:
                    print(f"    Reason: {report.ground_penetration.violation_reason}")
            else:
                # Normal motion, no output
                pass
        else:
            print(f"Failed to analyze {args.motion_file}")
    else:
        # Folder mode
        # Create annotations directory based on motion_folder
        project_root = Path(__file__).parent.parent.resolve()
        
        # Extract folder name from motion_folder path
        motion_folder_path = Path(args.motion_folder)
        folder_name = motion_folder_path.name if motion_folder_path.name else motion_folder_path.parts[-1] if motion_folder_path.parts else "motion_data"
        
        # Handle special characters
        folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
        folder_name = folder_name.strip('. ')
        if not folder_name:
            folder_name = "motion_data"
        
        # Create annotations directory
        annotations_dir = project_root / "annotations" / folder_name
        annotations_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotations will be saved to: {annotations_dir}")
        
        # Set progress_file path (if not specified)
        if args.progress_file is None:
            progress_file = str(annotations_dir / "auto_check_progress.json")
        else:
            progress_file = args.progress_file
        
        reports = scan_folder(
            args.motion_folder,
            args.robot_type,
            foot_names=foot_names,
            batch_size=args.batch_size,
            save_progress=not args.no_save_progress,
            progress_file=progress_file,
            resume_from_progress=not args.no_resume,  # If --no-resume, don't resume from progress
        )
        
        abnormal_reports = [r for r in reports if r.is_abnormal]
        
        print(f"\n{'='*80}")
        print(f"Check Results")
        print(f"{'='*80}")
        print(f"Total motions scanned: {len(reports)}")
        print(f"Abnormal motions detected: {len(abnormal_reports)}")
        
        # Prevent division by zero
        if len(reports) > 0:
            percentage = len(abnormal_reports) / len(reports) * 100
            print(f"Percentage: {percentage:.2f}%\n")
        else:
            print("Percentage: 0.00%\n")
        
        # Count by issue type
        self_collision_count = sum(1 for r in abnormal_reports if r.self_collision.has_self_collision)
        ground_penetration_count = sum(1 for r in abnormal_reports if r.ground_penetration.has_penetration)
        
        print(f"Self-collisions: {self_collision_count}")
        print(f"Ground penetration: {ground_penetration_count}\n")
        
        if abnormal_reports:
            print("Abnormal Motions:")
            print("-" * 80)
            for report in sorted(
                abnormal_reports,
                key=lambda r: (
                    r.self_collision.has_self_collision,
                    r.ground_penetration.has_penetration,
                ),
                reverse=True,
            ):
                issues = []
                if report.self_collision.has_self_collision:
                    issues.append("self_collision")
                if report.ground_penetration.has_penetration:
                    issues.append("ground_penetration")
                print(f"{report.motion_file}: {', '.join(issues)}")
        else:
            print("No abnormal motions detected.")
        
        # Save abnormal reports to JSON
        # annotations_dir was already created at the start of folder mode
        output_json = args.output_json
        if output_json is None and abnormal_reports:
            output_json = str(annotations_dir / "auto_check_reports.json")
        
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save only abnormal reports
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(r) for r in abnormal_reports],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nAbnormal reports ({len(abnormal_reports)} motions) saved to: {output_path}")
        
        # Save annotations format (only abnormal motions)
        annotations_path = annotations_dir / "auto_check_annotations.json"
        annotations = {}
        
        # Get project root (assuming script is in scripts/ directory)
        project_root = Path(__file__).parent.parent.resolve()
        
        # Get base path for motion_path (resolve to absolute path)
        motion_folder_path = (project_root / args.motion_folder).resolve()
        
        # Only annotate motions that have detected problems
        for report in reports:
            # Skip normal motions (no annotation)
            if not report.is_abnormal:
                continue
            
            # Generate reason string for abnormal motions
            issues = []
            if report.ground_penetration.has_penetration:
                issues.append("1. Interaction with terrain/scene")
            if report.self_collision.has_self_collision:
                issues.append("2. Clipping or self-collision")
            # Remove duplicates while preserving order
            seen = set()
            unique_issues = []
            for issue in issues:
                if issue not in seen:
                    seen.add(issue)
                    unique_issues.append(issue)
            reason = ", ".join(unique_issues) if unique_issues else "Other"
            
            # Get motion_path (relative to project root)
            motion_file_path = motion_folder_path / report.motion_file
            # Convert to relative path from project root
            try:
                motion_path = str(motion_file_path.relative_to(project_root))
            except ValueError:
                # If not relative to project root, use absolute path or keep as is
                motion_path = str(motion_file_path)
            
            # Use motion_file as key (same format as annotations.json)
            key = report.motion_file
            
            annotations[key] = {
                "label": "abnormal",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "motion_path": motion_path,
            }
        
        # Overwrite existing annotations (do not merge with old data)
        # Save annotations (only abnormal motions are annotated)
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(
                annotations,
                f,
                ensure_ascii=False,
                indent=2,
            )
        abnormal_count = sum(1 for v in annotations.values() if v.get("label") == "abnormal")
        print(f"\nAnnotations ({abnormal_count} abnormal motions) saved to: {annotations_path}")


if __name__ == "__main__":
    main()