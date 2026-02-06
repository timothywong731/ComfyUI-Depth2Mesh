from collections import defaultdict

import numpy as np
import trimesh
from PIL import Image


def depth2mesh(image_input, mesh_width_mm, mesh_height_mm, mesh_depth_mm, power=1.0):
    """
    Converts a heightmap (PNG path or PIL Image) to a closed mesh suitable for CNC milling.

    Pixels with alpha=0 are ignored. All walls and bottom are closed.

    Parameters:
    - image_input: Path to the heightmap PNG file OR a PIL Image object.
    - mesh_width_mm: Desired width of the mesh in millimeters (x-axis).
    - mesh_height_mm: Desired height of the mesh in millimeters (y-axis).
    - mesh_depth_mm: Desired maximum depth of the mesh in millimeters (z-axis).
    - power: Optional power transformation to apply to the depth values to make the z-axis more prominent.
             Default is 1.0 (no transformation).

    Returns:
    - A trimesh.Trimesh object representing the closed mesh.
    """
    # Step 1: Load the image
    # Handle both file paths (str) and PIL Image objects (from ComfyUI)
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGBA")
    else:
        image = image_input.convert("RGBA")

    width_px, height_px = image.size

    # Step 2: Create a mask based on the alpha channel
    # The alpha channel determines the "active" area of the mesh.
    # Pixels with alpha=0 are ignored and will not generate vertices.
    data = np.array(image)
    alpha = data[:, :, 3]
    mask = alpha != 0  # True for opaque pixels

    if not mask.any():
        raise ValueError("All pixels are transparent. No mesh to generate.")

    # Step 3: Compute the height map from the RGB channels (average for simplicity)
    # Convert RGB to grayscale intensity to represent height.
    rgb = data[:, :, :3]
    height_map = rgb.mean(axis=2).astype(np.float32)

    # Apply mask: set height to 0 where mask is False (outside the shape)
    height_map[~mask] = 0.0

    # Normalize the height map to range [0, 1]
    # This prepares the data for scaling by 'mesh_depth_mm' later.
    max_height_value = height_map.max()
    if max_height_value == 0:
        raise ValueError("Maximum height value is zero. Cannot scale depth.")
    normalized_height_map = height_map / max_height_value

    # Apply power transformation if provided to enhance z-axis prominence
    # Power > 1.0 accentuates peaks; Power < 1.0 flattens them.
    height_map_transformed = np.power(normalized_height_map, power)

    # Scale heights according to the desired mesh depth
    height_map_mm = height_map_transformed * mesh_depth_mm

    # Step 5: Create grid of x and y coordinates
    # Generate coordinate matrices for every pixel.
    xs = np.linspace(0, mesh_width_mm, width_px)
    ys = np.linspace(0, mesh_height_mm, height_px)
    xv, yv = np.meshgrid(xs, ys)

    # Flip y-axis to match image coordinates (optional, based on your coordinate system)
    # Images usually have (0,0) at top-left, but 3D world space often has Y up.
    yv = yv[::-1, :]

    # Flatten the arrays for processing
    # We will process vertices as a 1D list.
    xv = xv.flatten()
    yv = yv.flatten()
    zv = height_map_mm.flatten()

    # Flatten the mask
    mask_flat = mask.flatten()

    # Step 6: Keep only the vertices where mask is True (top vertices)
    # Filter out vertices that correspond to transparent pixels.
    top_vertices = np.column_stack((xv[mask_flat], yv[mask_flat], zv[mask_flat]))

    # Create a mapping from original grid to masked vertices
    # index_map[original_pixel_index] = new_vertex_index (or -1 if masked out)
    mask_indices = np.where(mask_flat)[0]
    index_map = -np.ones(width_px * height_px, dtype=int)
    index_map[mask_indices] = np.arange(len(mask_indices))

    # Step 7: Generate faces for the top mesh by connecting grid neighbors
    # Iterate through the grid and form 2 triangles (a quad) for every 2x2 block of valid pixels.
    faces = []
    for row in range(height_px - 1):
        for col in range(width_px - 1):
            idx = row * width_px + col
            idx_right = idx + 1
            idx_down = idx + width_px
            idx_diag = idx + width_px + 1

            # Check if all four corners are masked (valid)
            # We only generate faces if the entire quad is inside the mask.
            if (
                mask_flat[idx]
                and mask_flat[idx_right]
                and mask_flat[idx_down]
                and mask_flat[idx_diag]
            ):
                v0 = index_map[idx]
                v1 = index_map[idx_right]
                v2 = index_map[idx_down]
                v3 = index_map[idx_diag]

                # Add two triangles for the quad
                # Triangle 1: Top-Left -> Top-Right -> Bottom-Left
                # Triangle 2: Top-Right -> Bottom-Right -> Bottom-Left
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

    if len(faces) == 0:
        raise ValueError(
            "No faces were generated for the top mesh. Check the heightmap and mask."
        )

    faces = np.array(faces)
    top_mesh = trimesh.Trimesh(vertices=top_vertices, faces=faces, process=False)

    # Step 8: Create the bottom mesh by duplicating the top vertices with z=0
    # This creates the flat base of the object.
    bottom_vertices = top_mesh.vertices.copy()
    bottom_vertices[:, 2] = 0.0

    # Reverse the face winding for the bottom to ensure correct normals
    # Bottom faces should point downwards.
    bottom_faces = top_mesh.faces[:, [0, 2, 1]]
    bottom_mesh = trimesh.Trimesh(
        vertices=bottom_vertices, faces=bottom_faces, process=False
    )

    # Step 9: Find boundary edges manually
    def find_boundary_edges(faces):
        # Count how many times each edge is shared by a face.
        # Edges shared by 2 faces are internal; edges with count 1 are boundaries.
        edge_count = defaultdict(int)
        for face in faces:
            # Each face has three edges
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]])),
            ]
            for edge in edges:
                edge_count[edge] += 1
        # Boundary edges are those that appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    boundary_edges = find_boundary_edges(top_mesh.faces)

    if not boundary_edges:
        print("No boundary edges found; no walls will be generated.")
    else:
        print(f"Number of boundary edges: {len(boundary_edges)}")

    # Step 10: Generate wall meshes
    # Connect the boundary edges of the top mesh to the corresponding edges of the bottom mesh.
    wall_meshes = []
    for edge in boundary_edges:
        v0_top, v1_top = edge

        # Coordinates of top and bottom vertices
        v0_top_coord = top_mesh.vertices[v0_top]
        v1_top_coord = top_mesh.vertices[v1_top]
        v0_bottom_coord = bottom_mesh.vertices[v0_top]
        v1_bottom_coord = bottom_mesh.vertices[v1_top]

        # Define vertices for the wall quad
        wall_vertices = np.array(
            [v0_top_coord, v1_top_coord, v1_bottom_coord, v0_bottom_coord]
        )

        # Define two faces for the wall (2 triangles)
        wall_faces = np.array([[0, 1, 2], [0, 2, 3]])

        # Create wall mesh
        wall = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces, process=False)

        wall_meshes.append(wall)

    # Step 11: Concatenate all meshes
    meshes = [top_mesh, bottom_mesh] + wall_meshes
    combined_mesh = trimesh.util.concatenate(meshes)

    # Step 12: Debugging - Check vertex and face counts
    max_face_index = combined_mesh.faces.max()
    num_vertices = len(combined_mesh.vertices)
    print(f"Total vertices: {num_vertices}")
    print(f"Total faces: {len(combined_mesh.faces)}")
    print(f"Max face index: {max_face_index}")

    if max_face_index >= num_vertices:
        raise IndexError(
            f"Face index {max_face_index} out of bounds for vertices with size {num_vertices}."
        )

    # Step 13: Ensure the mesh is watertight
    if not combined_mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fill holes.")
        combined_mesh.fill_holes()

    # Step 14: Final processing to clean up the mesh
    try:
        combined_mesh.process(validate=True)
    except Exception as e:
        print(f"Error during mesh processing: {e}")
        # Optionally, skip processing or handle differently
        pass

    return combined_mesh
