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

                # Add two triangles for the quad (Counter-Clockwise winding for +Z normal)
                # Triangle 1: Top-Left -> Bottom-Left -> Top-Right
                # Triangle 2: Top-Right -> Bottom-Left -> Bottom-Right
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

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
        # Create a list of all directed edges in the top mesh
        directed_edges = []
        for face in faces:
            directed_edges.extend(
                [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            )

        # Count occurrences of geometric edges (ignoring direction)
        edge_count = defaultdict(int)
        for edge in directed_edges:
            edge_count[tuple(sorted(edge))] += 1

        # Boundary edges are those that appear only once geometrically.
        # We preserve their original direction from the face winding.
        boundary_edges = [
            edge for edge in directed_edges if edge_count[tuple(sorted(edge))] == 1
        ]
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
        # The quad order is v0_top -> v1_top -> v1_bottom -> v0_bottom
        # which satisfies CCW winding facing outwards.
        wall_vertices = np.array(
            [v0_top_coord, v1_top_coord, v1_bottom_coord, v0_bottom_coord]
        )

        # Define two triangles for the wall quad.
        # Skip degenerate triangles if top and bottom vertices coincide (z=0).
        wall_faces = []
        if not (
            np.allclose(v0_top_coord, v1_top_coord)
            or np.allclose(v1_top_coord, v1_bottom_coord)
            or np.allclose(v1_bottom_coord, v0_top_coord)
        ):
            wall_faces.append([0, 1, 2])

        if not (
            np.allclose(v0_top_coord, v1_bottom_coord)
            or np.allclose(v1_bottom_coord, v0_bottom_coord)
            or np.allclose(v0_bottom_coord, v0_top_coord)
        ):
            wall_faces.append([0, 2, 3])

        if wall_faces:
            wall = trimesh.Trimesh(
                vertices=wall_vertices, faces=wall_faces, process=False
            )
            wall_meshes.append(wall)

    # Step 11: Concatenate all meshes
    meshes = [top_mesh, bottom_mesh] + wall_meshes
    combined_mesh = trimesh.util.concatenate(meshes)

    # Step 12: Consolidate geometry
    # Merge vertices that are at the same location to connect the disparate parts.
    combined_mesh.merge_vertices()

    # Remove any degenerate or duplicate faces.
    combined_mesh.update_faces(combined_mesh.nondegenerate_faces())
    combined_mesh.update_faces(combined_mesh.unique_faces())

    # Remove vertices that are no longer used after face removal.
    combined_mesh.remove_unreferenced_vertices()

    # Step 13: Ensure the mesh is watertight
    if not combined_mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fill holes.")
        try:
            combined_mesh.fill_holes()
        except Exception as e:
            print(f"Failed to fill holes: {e}")

    # Step 14: Final processing to clean up the mesh
    try:
        # process(validate=True) checks for watertightness and fixes normals.
        combined_mesh.process(validate=True)
    except Exception as e:
        print(f"Error during mesh processing: {e}")
        pass

    return combined_mesh
