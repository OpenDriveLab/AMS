#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read annotations from annotations_latest.json, classify by label and merge corresponding pkl files

Features:
1. Read annotation data from annotations_latest.json
2. Classify motion_path by label
3. Find corresponding pkl files based on specified motions subdirectory
4. Merge and save motion data for each label to merged_motions/{subdirectory_name}/ directory
"""
# python process_annotations_to_pkl.py -a annotations_latest.json --motion_source AMASS --save_name AMASS_29dof

import os
import json
import joblib
import argparse
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


def load_annotations(annotation_file):
    """Load annotation file"""
    print(f"Loading annotation file: {annotation_file}")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotation entries")
    return annotations


def path_to_key(file_path, base_dir=None):
    """
    Convert file path to key name
    Example: motions/g1/sampled_static_poses/GRAB/s1/airplane_fly_1_stageii.pkl 
    -> g1_sampled_static_poses_GRAB_s1_airplane_fly_1_stageii
    
    Args:
        file_path: File path (can be absolute or relative)
        base_dir: Base directory to remove from path (optional, if None, uses project root)
    """
    # Get absolute path
    if os.path.isabs(file_path):
        abs_path = file_path
    else:
        abs_path = os.path.abspath(file_path)
    
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    
    # Convert to relative path from project root
    try:
        rel_path = os.path.relpath(abs_path, project_root)
    except ValueError:
        # If not relative to project root, use the original path
        rel_path = str(file_path)
    
    # Remove base_dir prefix if specified
    if base_dir:
        if rel_path.startswith(base_dir + "/"):
            rel_path = rel_path[len(base_dir) + 1:]
        elif rel_path.startswith(base_dir):
            rel_path = rel_path[len(base_dir):]
    
    # Remove .pkl suffix
    if rel_path.endswith(".pkl"):
        rel_path = rel_path[:-4]
    
    # Replace / with _
    key_name = rel_path.replace("/", "_")
    
    return key_name


def convert_motion_path(motion_path, target_dir):
    """
    Convert motion_path to path under target directory
    
    Args:
        motion_path: Original path, e.g., "data/out/GRAB/s1/airplane_fly_1_stageii.pkl" or "motions/g1/sampled_static_poses/test/collision_example.pkl"
        target_dir: Target directory path (relative to project root), e.g., "motions/g1/sampled_static_poses"
    
    Returns:
        Converted path relative to project root, e.g., "motions/g1/sampled_static_poses/GRAB/s1/airplane_fly_1_stageii.pkl"
    """
    # If motion_path already starts with target_dir, use it directly (no conversion needed)
    if motion_path.startswith(target_dir + "/") or motion_path == target_dir:
        return motion_path
    
    # Extract the relative part from motion_path
    # Handle different path formats
    if motion_path.startswith("data/out/"):
        # Extract path after data/out/
        rel_part = motion_path[len("data/out/"):]
        new_path = f"{target_dir}/{rel_part}"
    elif motion_path.startswith("motions/"):
        # Extract path after motions/{subdir}/
        # Split into parts: ["motions", "{subdir}", "{rest}"]
        parts = motion_path.split("/", 2)
        if len(parts) >= 3:
            # Use the part after motions/{subdir}/
            new_path = f"{target_dir}/{parts[2]}"
        else:
            # If format is just "motions/xxx.pkl", use target_dir directly
            new_path = f"{target_dir}/{motion_path.split('/', 1)[1]}" if '/' in motion_path else f"{target_dir}/{motion_path}"
    else:
        # If path format doesn't match expected, try direct concatenation
        new_path = f"{target_dir}/{motion_path}"
    
    return new_path


def group_by_label(annotations, target_dir):
    """
    Classify motion_path by label and convert to paths under target directory
    
    Args:
        annotations: Annotation data dictionary
        target_dir: Target directory path (relative to project root)
    
    Returns:
        dict: {label: [motion_paths]}
    """
    label_paths = defaultdict(list)
    
    for key, item in annotations.items():
        label = item.get('label')
        motion_path = item.get('motion_path')
        
        if not label or not motion_path:
            continue
        
        # Convert to path under target directory
        new_path = convert_motion_path(motion_path, target_dir)
        label_paths[label].append(new_path)
    
    return label_paths


def get_annotated_paths(annotations, target_dir):
    """
    Get set of all annotated file paths
    
    Args:
        annotations: Annotation data dictionary
        target_dir: Target directory path (relative to project root)
    
    Returns:
        set: Set of annotated file paths
    """
    annotated_paths = set()
    
    for key, item in annotations.items():
        motion_path = item.get('motion_path')
        if motion_path:
            # Convert to path under target directory
            new_path = convert_motion_path(motion_path, target_dir)
            annotated_paths.add(new_path)
    
    return annotated_paths


def find_all_pkl_files(target_dir):
    """
    Recursively find all pkl files in target directory
    
    Args:
        target_dir: Target directory path
    
    Returns:
        list: List of absolute paths of all pkl files
    """
    pkl_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return sorted(pkl_files)


def find_unlabeled_files(target_dir_abs, annotated_paths, project_root):
    """
    Find unlabeled pkl files
    
    Args:
        target_dir_abs: Target directory absolute path
        annotated_paths: Set of annotated file paths (relative to project root)
        project_root: Project root directory path
    
    Returns:
        list: List of unlabeled pkl file absolute paths
    """
    all_pkl_files = find_all_pkl_files(target_dir_abs)
    unlabeled_files = []
    
    # Convert annotated_paths to absolute path set for comparison
    annotated_abs_paths = set()
    for annotated_path in annotated_paths:
        # annotated_path is relative to project root
        if os.path.isabs(annotated_path):
            abs_path = annotated_path
        else:
            abs_path = os.path.join(project_root, annotated_path)
        annotated_abs_paths.add(os.path.abspath(abs_path))
    
    for pkl_file in all_pkl_files:
        # pkl_file is already an absolute path
        pkl_file_abs = os.path.abspath(pkl_file)
        
        if pkl_file_abs not in annotated_abs_paths:
            unlabeled_files.append(pkl_file)
    
    return unlabeled_files


def process_unlabeled_files(unlabeled_paths, target_dir, output_base_dir, save_name):
    """
    Process unlabeled files and save
    
    Args:
        unlabeled_paths: List of unlabeled file paths
        target_dir: Target directory path (relative to project root)
        output_base_dir: Output base directory
        save_name: Save path name
    
    Returns:
        dict: Processing result statistics
    """
    if not unlabeled_paths:
        print("\nNo unlabeled files found")
        return None
    
    print(f"\nFound {len(unlabeled_paths)} unlabeled files")
    
    merged_data = {}
    failed_files = []
    
    for pkl_file in tqdm(unlabeled_paths, desc="Loading unlabeled data"):
        try:
            # Load pkl file
            data = joblib.load(pkl_file)
            
            # Generate key name (relative to project root)
            key_name = path_to_key(pkl_file, target_dir)
            
            # If data is a dict with only one key, extract inner data
            if isinstance(data, dict) and len(data) == 1:
                inner_key = list(data.keys())[0]
                data = data[inner_key]
            
            # Check if key already exists
            if key_name in merged_data:
                print(f"\nWarning: key '{key_name}' already exists, will be overwritten")
            
            merged_data[key_name] = data
            
        except Exception as e:
            print(f"\nError: Failed to read file {pkl_file}: {e}")
            failed_files.append(pkl_file)
    
    if not merged_data:
        print("No unlabeled files were successfully loaded")
        return None
    
    # Save unlabeled data
    output_dir = os.path.join(output_base_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{save_name}_motions_unlabeled.pkl")
    
    print(f"\nSaving unlabeled data to: {output_file}")
    joblib.dump(merged_data, output_file)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    print(f"Number of entries: {len(merged_data)}")
    
    result = {
        'data': merged_data,
        'total': len(unlabeled_paths),
        'found': len(merged_data),
        'failed': len(failed_files),
        'failed_files': failed_files[:10],
        'output_file': output_file,
        'file_size': file_size
    }
    
    return result


def load_and_merge_pkl_files(label_paths, target_dir, project_root):
    """
    Load and merge pkl files based on path list
    
    Args:
        label_paths: {label: [motion_paths]} (paths relative to project root)
        target_dir: Target directory path (relative to project root)
        project_root: Project root directory path
    
    Returns:
        dict: {label: merged_data_dict}
    """
    results = {}
    
    for label, paths in label_paths.items():
        print(f"\nProcessing label: {label}")
        print(f"Contains {len(paths)} paths")
        
        merged_data = {}
        missing_files = []
        failed_files = []
        
        for motion_path in tqdm(paths, desc=f"Loading {label} data"):
            # Convert relative path to absolute path
            if os.path.isabs(motion_path):
                abs_motion_path = motion_path
            else:
                abs_motion_path = os.path.join(project_root, motion_path)
            
            # Check if file exists
            if not os.path.exists(abs_motion_path):
                missing_files.append(motion_path)
                continue
            
            try:
                # Load pkl file
                data = joblib.load(abs_motion_path)
                
                # Generate key name (relative to target_dir)
                key_name = path_to_key(abs_motion_path, target_dir)
                
                # If data is a dict with only one key, extract inner data
                if isinstance(data, dict) and len(data) == 1:
                    inner_key = list(data.keys())[0]
                    data = data[inner_key]
                
                # Check if key already exists
                if key_name in merged_data:
                    print(f"\nWarning: key '{key_name}' already exists, will be overwritten")
                
                merged_data[key_name] = data
                
            except Exception as e:
                print(f"\nError: Failed to read file {motion_path}: {e}")
                failed_files.append(motion_path)
        
        results[label] = {
            'data': merged_data,
            'total': len(paths),
            'found': len(merged_data),
            'missing': len(missing_files),
            'failed': len(failed_files),
            'missing_files': missing_files[:10],  # Only save first 10 for display
            'failed_files': failed_files[:10]
        }
        
        print(f"Successfully loaded: {len(merged_data)}/{len(paths)} files")
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
        if failed_files:
            print(f"Failed files: {len(failed_files)}")
    
    return results


def save_merged_data(results, output_base_dir, save_name):
    """
    Save merged data to files
    
    Args:
        results: {label: {data, ...}}
        output_base_dir: Output base directory, e.g., "merged_motions"
        save_name: Save path name
    """
    # Create output directory
    output_dir = os.path.join(output_base_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving data to directory: {output_dir}")
    
    saved_files = {}
    
    for label, result in results.items():
        output_file = os.path.join(output_dir, f"{save_name}_motions_{label}.pkl")
        
        print(f"\nSaving {label} data to: {output_file}")
        joblib.dump(result['data'], output_file)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        saved_files[label] = {
            'file': output_file,
            'size': file_size,
            'count': len(result['data'])
        }
        
        print(f"File size: {file_size:.2f} MB")
    
    return saved_files


def print_summary(label_paths, results, saved_files, unlabeled_result=None):
    """Print processing summary"""
    print("\n" + "=" * 60)
    print("Processing complete! Statistics:")
    print("=" * 60)
    
    for label in sorted(label_paths.keys()):
        result = results[label]
        saved = saved_files[label]
        
        print(f"\n{label}:")
        print(f"  - Label paths: {result['total']}")
        print(f"  - Successfully loaded: {result['found']}")
        print(f"  - Missing files: {result['missing']}")
        print(f"  - Failed files: {result['failed']}")
        print(f"  - Output file: {saved['file']}")
        print(f"  - File size: {saved['size']:.2f} MB")
        print(f"  - Number of entries: {saved['count']}")
        
        if result['missing_files']:
            print(f"  - First 10 missing files:")
            for f in result['missing_files']:
                print(f"      {f}")
        
        if result['failed_files']:
            print(f"  - First 10 failed files:")
            for f in result['failed_files']:
                print(f"      {f}")
    
    # Print unlabeled files statistics
    if unlabeled_result:
        print(f"\nUnlabeled files (unlabeled):")
        print(f"  - Unlabeled files: {unlabeled_result['total']}")
        print(f"  - Successfully loaded: {unlabeled_result['found']}")
        print(f"  - Failed files: {unlabeled_result['failed']}")
        print(f"  - Output file: {unlabeled_result['output_file']}")
        print(f"  - File size: {unlabeled_result['file_size']:.2f} MB")
        print(f"  - Number of entries: {unlabeled_result['found']}")
        
        if unlabeled_result['failed_files']:
            print(f"  - First 10 failed files:")
            for f in unlabeled_result['failed_files']:
                print(f"      {f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Read annotations from annotations_latest.json, classify by label and merge corresponding pkl files'
    )
    
    parser.add_argument(
        '-a', '--annotations',
        type=str,
        default='annotations_latest.json',
        help='Annotation file path (default: annotations_latest.json)'
    )
    
    parser.add_argument(
        '--motion_source',
        type=str,
        required=True,
        help='Data source directory path (relative to project root), e.g., "motions/g1/sampled_static_poses"'
    )
    
    parser.add_argument(
        '--save_name',
        type=str,
        required=True,
        help='Save path name, used to specify output directory (merged_motions/{save_name}) and filename, e.g., AMASS, etc.'
    )
    
    parser.add_argument(
        '-o', '--output_base',
        type=str,
        default='merged_motions',
        help='Output base directory (default: merged_motions)'
    )
    
    args = parser.parse_args()
    
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.resolve()
    
    # Check if annotation file exists (relative to project root)
    annotation_path = args.annotations
    if not os.path.isabs(annotation_path):
        annotation_path = os.path.join(project_root, annotation_path)
    
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file {annotation_path} does not exist")
        return
    
    # Use motion_source directly as target directory (relative to project root)
    target_dir = args.motion_source
    
    # Convert to absolute path for file operations
    if os.path.isabs(target_dir):
        target_dir_abs = target_dir
    else:
        target_dir_abs = os.path.join(project_root, target_dir)
    
    if not os.path.exists(target_dir_abs):
        print(f"Error: Target directory {target_dir_abs} does not exist")
        print(f"Please check if the directory exists relative to project root: {project_root}")
        return
    
    # Output directory
    output_dir = os.path.join(args.output_base, args.save_name)
    
    print(f"Project root: {project_root}")
    print(f"Input annotation file: {annotation_path}")
    print(f"Data source directory (relative): {target_dir}")
    print(f"Data source directory (absolute): {target_dir_abs}")
    print(f"Save path name: {args.save_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 1. Load annotation file
    annotations = load_annotations(annotation_path)
    
    # 2. Classify motion_path by label
    print("\nClassifying motion_path by label...")
    label_paths = group_by_label(annotations, target_dir)
    
    print(f"\nFound {len(label_paths)} different labels:")
    for label, paths in sorted(label_paths.items()):
        print(f"  - {label}: {len(paths)} paths")
    
    # 3. Load and merge pkl files
    print("\nLoading and merging pkl files...")
    results = load_and_merge_pkl_files(label_paths, target_dir, project_root)
    
    # 4. Save merged data
    saved_files = save_merged_data(results, args.output_base, args.save_name)
    
    # 5. Find and process unlabeled files
    print("\n" + "=" * 60)
    print("Finding unlabeled files...")
    print("=" * 60)
    
    # Get all annotated file paths
    annotated_paths = get_annotated_paths(annotations, target_dir)
    print(f"Number of annotated files: {len(annotated_paths)}")
    
    # Find unlabeled files
    unlabeled_paths = find_unlabeled_files(target_dir_abs, annotated_paths, project_root)
    
    # Process unlabeled files
    unlabeled_result = process_unlabeled_files(
        unlabeled_paths, 
        target_dir, 
        args.output_base, 
        args.save_name
    )
    
    # 6. Print summary
    print_summary(label_paths, results, saved_files, unlabeled_result)


if __name__ == '__main__':
    main()

