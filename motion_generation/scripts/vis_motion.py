import argparse
import time
import numpy as np
import yourdfpy
import viser
from viser.extras import ViserUrdf
import joblib

def load_and_normalize_motion(file_path):
    # Load raw data
    data = joblib.load(file_path)
    
    sequences = {}
    
    if isinstance(data, dict):

        if 'root_trans_offset' in data and 'root_rot' in data and 'dof' in data:
            print("safe")
            print("Detected MotionGen format - converting to unified qpos format...")
            pos = np.array(data['root_trans_offset'])  # (N, 3)
            rot = np.array(data['root_rot'])          # (N, 4) - [w, x, y, z]
            dof = np.array(data['dof'])                # (N, num_joints)
            qpos = np.concatenate([pos, rot, dof], axis=1)  # (N, 3+4+num_joints)
            sequences["converted_sequence"] = qpos
            print(f"Converted to qpos array of shape {qpos.shape}")
        else:
            for name, value in data.items():
                # Unwrap nested 0-dim arrays
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    value = value.item()
                
                if isinstance(value, dict) and 'root_trans_offset' in value:
                    print(f"Converting sequence '{name}'...")
                    pos = np.array(value['root_trans_offset'])
                    rot = np.array(value['root_rot'])
                    dof = np.array(value['dof'])
                    sequences[name] = np.concatenate([pos, rot, dof], axis=1)
                elif isinstance(value, np.ndarray) and value.ndim == 2:
                    sequences[name] = value
                else:
                    print(f"Skipping unrecognized format for sequence '{name}' (type: {type(value)})")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    if not sequences:
        raise ValueError("No valid motion sequences found after normalization")
    
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Visualize G1 motion sequence from .pkl files.")
    parser.add_argument("file_path", type=str, help="Path to motion sequence file")
    parser.add_argument(
        "--urdf",
        type=str,
        default="robot/unitree_description/urdf/g1/g1_sysid_29dof.urdf",
        help="Path to robot URDF"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    args = parser.parse_args()

    # Load and normalize motion data
    sequences = load_and_normalize_motion(args.file_path)
    print(f"Loaded {len(sequences)} sequence(s): {list(sequences.keys())}")

    # Get first sequence
    first_name = list(sequences.keys())[0]
    qpos_seq = sequences[first_name]
    num_steps = qpos_seq.shape[0]
    print(f"Using sequence '{first_name}' with {num_steps} frames (shape: {qpos_seq.shape})")

    # Setup visualization
    print(f"Loading URDF: {args.urdf}")
    urdf = yourdfpy.URDF.load(args.urdf)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, position=(0, 0, -0.003))
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # GUI controls
    with server.gui.add_folder("Playback Control"):
        seq_dropdown = server.gui.add_dropdown(
            "Sequence", 
            options=list(sequences.keys()), 
            initial_value=first_name
        )
        playing = server.gui.add_checkbox("Playing", True)
        loop = server.gui.add_checkbox("Loop", True)
        fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=args.fps)
        frame_slider = server.gui.add_slider("Frame", min=0, max=num_steps - 1, step=1, initial_value=0)

    # State management
    state = {"qpos": qpos_seq, "num_steps": num_steps, "name": first_name}

    @seq_dropdown.on_update
    def _(_):
        name = seq_dropdown.value
        new_seq = sequences[name]
        state.update({"qpos": new_seq, "num_steps": new_seq.shape[0], "name": name})
        frame_slider.max = new_seq.shape[0] - 1
        frame_slider.value = 0
        print(f"Switched to '{name}' ({new_seq.shape[0]} frames)")

    # Animation loop
    last_update = time.time()
    while True:
        current_time = time.time()
        dt = 1.0 / fps_slider.value
        
        # Auto-advance frame
        if playing.value and (current_time - last_update) >= dt:
            next_frame = frame_slider.value + 1
            max_frame = state["num_steps"] - 1
            
            if next_frame > max_frame:
                if loop.value:
                    next_frame = 0
                else:
                    next_frame = max_frame
                    playing.value = False
            
            frame_slider.value = next_frame
            last_update = current_time
        
        t = min(frame_slider.value, state["num_steps"] - 1)
        frame_data = state["qpos"][t]
        
        
        # Extract components: [pos(3), quat(4), joints(N)]
        try:
            pos = frame_data[:3]
            quat = frame_data[3:7][[3, 0, 1, 2]]
            joints = frame_data[7:]
            
            with server.atomic():
                base_frame.position = np.array(pos)
                base_frame.wxyz = np.array(quat)
                urdf_vis.update_cfg(np.array(joints))
        except Exception as e:
            print(f"Frame {t} rendering error: {e}")
        
        time.sleep(0.005)

if __name__ == "__main__":
    main()