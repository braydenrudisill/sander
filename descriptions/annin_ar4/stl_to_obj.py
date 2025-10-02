import open3d as o3d
import os

def convert_stl_to_obj_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".stl"):
            stl_path = os.path.join(input_dir, filename)
            obj_filename = os.path.splitext(filename)[0] + ".obj"
            obj_path = os.path.join(output_dir, obj_filename)

            try:
                mesh = o3d.io.read_triangle_mesh(stl_path)
                o3d.io.write_triangle_mesh(obj_path, mesh)
                print(f"Converted '{filename}' to '{obj_filename}'")
            except Exception as e:
                print(f"Error converting '{filename}': {e}")

# Example usage:
input_directory = "meshes/ar4_gripper_stl"  # Replace with your input directory
output_directory = "meshes/ar4_gripper" # Replace with your output directory
convert_stl_to_obj_directory(input_directory, output_directory)