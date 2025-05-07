import open3d as o3d
import numpy as np
import copy

import open3d as o3d
import numpy as np
import time 
import copy

DEBUG=True


import open3d as o3d
import numpy as np
import copy

def check_gripper_collision(point_cloud_path, gripper_obj_path, position, rotation_matrix, 
                           scale_factor=0.01, collision_threshold=0.005, visualize=False, floor_height=None, 
                           enforce_floor_constraint=False):
    """
    Check if a gripper model at a given position and orientation collides with a point cloud.
    
    Parameters:
    - point_cloud_path: Path to the point cloud PLY file
    - gripper_obj_path: Path to the gripper OBJ file
    - position: 3D position [x, y, z] for the gripper
    - rotation_matrix: 3x3 rotation matrix for the gripper
    - scale_factor: Scaling factor to apply to the gripper model (0.01 for mm to m)
    - collision_threshold: Distance threshold for collision detection (meters)
    - visualize: Boolean, whether to visualize the result
    
    Returns:
    - collision: Boolean, True if collision detected
    - min_distance: Minimum distance between gripper and point cloud
    """
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    
    # Load the gripper model from OBJ
    gripper_mesh = o3d.io.read_triangle_mesh(gripper_obj_path)
    
    # Ensure the mesh has vertex normals for better visualization
    if not gripper_mesh.has_vertex_normals():
        gripper_mesh.compute_vertex_normals()
    
    # Scale the gripper model (before transformation)
    gripper_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(gripper_mesh.vertices) * scale_factor
    )
    
    if enforce_floor_constraint and floor_height is None:
        pc_points = np.asarray(point_cloud.points)
        floor_height = np.max(pc_points[:, 2]) - 0.01 

    # Create 4x4 transformation matrix from position and rotation
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position
    
    # Apply transformation to the gripper
    gripper_transformed = copy.deepcopy(gripper_mesh)
    gripper_transformed.transform(transformation)
    
    # Convert the gripper mesh to a point cloud for distance calculation
    gripper_pcd = gripper_transformed.sample_points_uniformly(number_of_points=1000)
    gripper_points = np.asarray(gripper_pcd.points)
    
    
    # Build a KD tree for the scene point cloud for efficient distance queries
    scene_tree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # Check for collisions by computing distances from gripper points to scene
    collision = False
    min_distance = float('inf')
    collision_points = []
    
    floor_collision = False

    if enforce_floor_constraint:
        # Check if any gripper points are below the floor
        gripper_heights = gripper_points[:, 2]
        points_below_floor = gripper_heights > floor_height
        
        if np.any(points_below_floor):
            floor_collision = True
            # floor_penetration_depth = floor_height - np.min(gripper_heights)

    # Build a KD tree for the scene point cloud for efficient distance queries
    scene_tree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # Check for collisions by computing distances from gripper points to scene
    pc_collision = False
    min_distance = float('inf')
    collision_points = []
    
    for point in gripper_points:
        # Find the closest point in the scene to this gripper point
        k, idx, squared_distances = scene_tree.search_knn_vector_3d(point, 1)
        distance = np.sqrt(squared_distances[0])
        
        min_distance = min(min_distance, distance)
        
        # Check if the distance is below the collision threshold
        if distance < collision_threshold:
            pc_collision = True
            collision_points.append(point)
            if not visualize and not enforce_floor_constraint:
                break  # Early termination if visualization not needed
    
    # Determine collision type
    collision = pc_collision or floor_collision
    collision_type = None
    if pc_collision:
        collision_type = 'point_cloud'
    elif floor_collision:
        collision_type = 'floor'

    # Visualization
    if visualize:
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Gripper Collision Check", width=1024, height=768)
        
        # Add point cloud
        vis.add_geometry(point_cloud)
        
        # Color the gripper based on collision
        if collision:
            if collision_type == 'point_cloud':
                gripper_transformed.paint_uniform_color([1, 0, 0])  # Red for point cloud collision
            else:
                gripper_transformed.paint_uniform_color([1, 0.5, 0])  # Orange for floor collision
        else:
            gripper_transformed.paint_uniform_color([0, 1, 0])  # Green for no collision
        
        vis.add_geometry(gripper_transformed)
        
        # Add coordinate frame at gripper position
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=position)
        vis.add_geometry(coord_frame)
        
        # If point cloud collision points exist, visualize them
        if pc_collision and collision_points:
            collision_pcd = o3d.geometry.PointCloud()
            collision_pcd.points = o3d.utility.Vector3dVector(np.array(collision_points))
            collision_pcd.paint_uniform_color([1, 1, 0])  # Yellow
            vis.add_geometry(collision_pcd)
        
        # Visualize the floor plane
        if enforce_floor_constraint:
            # Create floor plane geometry
            bbox = point_cloud.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            
            # Make the floor a bit larger than the point cloud
            margin = 1.0  # 1 meter margin
            floor_min_x = min_bound[0] - margin
            floor_min_y = min_bound[1] - margin
            floor_max_x = max_bound[0] + margin
            floor_max_y = max_bound[1] + margin
            
            # Create a plane
            floor_points = [
                [floor_min_x, floor_min_y, floor_height],
                [floor_max_x, floor_min_y, floor_height],
                [floor_max_x, floor_max_y, floor_height],
                [floor_min_x, floor_max_y, floor_height]
            ]
            
            # Create triangles for the mesh
            floor_triangles = [
                [0, 1, 2],  # First triangle
                [0, 2, 3]   # Second triangle
            ]
            
            floor_mesh = o3d.geometry.TriangleMesh()
            floor_mesh.vertices = o3d.utility.Vector3dVector(floor_points)
            floor_mesh.triangles = o3d.utility.Vector3iVector(floor_triangles)
            floor_mesh.compute_vertex_normals()
            floor_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
            
            # Make semi-transparent
            floor_mesh.compute_triangle_normals()
            vis.add_geometry(floor_mesh)
        
        # Optimize view
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 3.0
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    # Return collision status, distance, and collision type
    return collision, min_distance, collision_type

def test_gripper_placement(point_cloud_path, gripper_obj_path, scale_factor=0.01):
    """
    Test the gripper at different positions in the scene to find potential grasp points.
    """
    # Load point cloud
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    
    # Get point cloud bounding box and center
    bbox = point_cloud.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    
    # Adjust the center to match point cloud Z height (appears to be raised)
    # Your point cloud center is at ~9m in Z, so we should adjust positioning
    center[2] = 9.0  # Adjust based on your point cloud center
    
    # Create example positions to test
    # We'll create a 3x3 grid at the center height of the point cloud
    positions = []
    for x_offset in [-1.0, 1.0]:
        for y_offset in [-1.0, 1.0]:
            for z_offset in [-2.0, -1.0, 0.0]:
                # Place the gripper at different points around the center
                positions.append([
                    center[0] + x_offset,
                    center[1] + y_offset,
                    center[2] + z_offset
                ])
        
    # Create some rotation matrices to test (horizontal and vertical approaches)
    rotations = []
    
    # Rotate 90 degrees around X (approach from top)
    rot_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    rotations.append(rot_x_90)
    
    # Test each position and rotation
    results = []
    entire_process_time_start = time.time()
    for pos in positions:
        for rot in rotations:
            start = time.time()
            collision, distance, collision_type = check_gripper_collision(
                point_cloud_path,
                gripper_obj_path,
                np.array(pos),
                rot,
                scale_factor=scale_factor,
                collision_threshold=0.02,  # 2cm
                visualize=False,
                enforce_floor_constraint=True,
            )
            end = time.time()
            check_time = end-start
            
            results.append({
                "position": pos,
                "rotation": rot,
                "collision": collision,
                "distance": distance
            })
            
            print(f"Position: {pos}, Collision: {collision}, Distance: {distance:.4f}m, Collision_type: {collision_type}, Check time: {check_time}")
    entire_process_time_end = time.time()
    entire_process_time = entire_process_time_end - entire_process_time_start
    print('Entire process tooks: ', entire_process_time)
    # Find best non-colliding position
    valid_positions = [r for r in results if not r["collision"]]
    if valid_positions:
        best = min(valid_positions, key=lambda x: x["distance"])
        print("\nBest non-colliding position found:")
        print(f"Position: {best['position']}")
        print(f"Distance: {best['distance']:.4f}m")
        
        # Visualize the best position again
        check_gripper_collision(
            point_cloud_path,
            gripper_obj_path,
            np.array(best["position"]),
            best["rotation"],
            scale_factor=scale_factor,
            collision_threshold=0.02,
            visualize=True
        )
    else:
        print("No valid positions found. Try adjusting parameters.")

# Run the test with appropriate scale (mm to m)
test_gripper_placement(
    "/home/elena/repos/Depth-Anything-V2-for-Robotic-Picking/metric_depth/vis_pointcloud/rgb.ply",
    "/home/elena/repos/Depth-Anything-V2-for-Robotic-Picking/gripperv3.obj", 
    scale_factor=0.01  # Convert mm to m
)