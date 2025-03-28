import os
import random
import glob
from pathlib import Path

def create_train_val_split(dataset_path, output_dir, train_ratio=0.8):
    """
    Creates train.txt and val.txt files with paired RGB and depth map paths.
    
    Args:
        dataset_path: Root path of the dataset containing rgb and depth_exr folders
        output_dir: Directory where train.txt and val.txt will be saved
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB image paths
    rgb_path = os.path.join(dataset_path, 'rgb')
    print(rgb_path)
    rgb_files = []
    
    # Walk through all subdirectories to find all RGB images
    for root, _, files in os.walk(rgb_path):
        for file in files:
            # print(file)
            if file.endswith(('.jpg', '.png', '.jpeg')):
                rgb_files.append(os.path.join(root, file))
    print(len(rgb_files))
    # For each RGB file, find the corresponding depth file
    paired_files = []
    for rgb_file in rgb_files:
        # Get relative path from the rgb directory
        rel_path = os.path.relpath(rgb_file, rgb_path)
        # Get directory structure
        dir_structure = os.path.dirname(rel_path)
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(rgb_file))[0]
        
        # Assume depth file has same name structure but replace "rgb" with "depth"
        # Get the rgb part of the filename (like rgb_00560)
        filename_parts = base_name.split('_')
        depth_name = 'depth_' + filename_parts[-1] if len(filename_parts) > 1 else f"{base_name}"
        print(depth_name)
        
        # Construct depth file path (replacing 'rgb' directory with 'depth_exr')
        depth_dir = os.path.join(dataset_path, 'depth_exr', dir_structure)
        print(depth_dir)
        
        # Look for the depth file with various possible extensions
        depth_file = None
        for ext in ['.exr', '.png', '.jpg', '.jpeg']:
            possible_depth = os.path.join(depth_dir, f"{depth_name}{ext}")
            print(possible_depth)
            print(os.path.exists(possible_depth))
            if os.path.exists(possible_depth):
                depth_file = possible_depth
                break
        
        # If depth file exists, add the pair to our list
        if depth_file:
            paired_files.append((rgb_file, depth_file))
    
    # Shuffle the paired files
    random.shuffle(paired_files)
    
    # Split into training and validation sets
    split_idx = int(len(paired_files) * train_ratio)
    train_pairs = paired_files[:split_idx]
    val_pairs = paired_files[split_idx:]
    
    # Write to train.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for rgb_file, depth_file in train_pairs:
            f.write(f"{os.path.abspath(rgb_file)} {os.path.abspath(depth_file)}\n")
    
    # Write to val.txt
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for rgb_file, depth_file in val_pairs:
            f.write(f"{os.path.abspath(rgb_file)} {os.path.abspath(depth_file)}\n")
    
    print(f"Created train.txt with {len(train_pairs)} pairs and val.txt with {len(val_pairs)} pairs")
    print(f"Files saved in: {os.path.join(output_dir, 'val.txt')}")

# Example usage
if __name__ == "__main__":
    dataset_path = "/path/to/dataset"  # Replace with your dataset path
    output_dir = "metric_depth/dataset/splits/cem/"      # Replace with your output directory
    
    create_train_val_split(dataset_path, output_dir)