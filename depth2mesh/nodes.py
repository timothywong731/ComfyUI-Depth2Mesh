import numpy as np
from PIL import Image

from .core import depth2mesh


class DepthMapToMesh:
    """
    ComfyUI Node to convert a Depth Map image into a 3D Mesh (STL).
    """

    @classmethod
    def INPUT_TYPES(s):
        # Define input parameters for the node
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The depth map image to convert."}),
                "width_mm": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.1,
                        "max": 10000.0,
                        "step": 0.1,
                        "tooltip": "Physical stock width in millimeters.",
                    },
                ),
                "height_mm": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.1,
                        "max": 10000.0,
                        "step": 0.1,
                        "tooltip": "Physical stock height in millimeters.",
                    },
                ),
                "depth_mm": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.1,
                        "max": 1000.0,
                        "step": 0.1,
                        "tooltip": "Maximum thickness of the stock in millimeters.",
                    },
                ),
                "power": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Exponent for depth curve. >1 makes peaks spikier, <1 makes them flatter.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate"
    CATEGORY = "DepthMap2STL"

    def generate(self, image, width_mm, height_mm, depth_mm, power):
        # ComfyUI provides images as [Batch, Height, Width, Channels] tensors in range 0-1.
        # We need to convert the first image in the batch to a PIL Image (0-255 uint8)
        # for our core processing logic.
        img_np = image[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8))

        # Call the core conversion logic
        mesh = depth2mesh(img_pil, width_mm, height_mm, depth_mm, power)
        return (mesh,)


class SimplifyMesh:
    """
    ComfyUI Node to reduce the face count of a mesh.
    Useful for reducing file size and processing time.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "The mesh to simplify."}),
                "target_face_count": (
                    "INT",
                    {
                        "default": 100000,
                        "min": 100,
                        "max": 10000000,
                        "step": 100,
                        "tooltip": "Target number of faces for the simplified mesh.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "simplify"
    CATEGORY = "DepthMap2STL"

    def simplify(self, mesh, target_face_count):
        # We use trimesh's simplify_quadric_decimation.
        # Note: 'fast-simplification' python package is highly recommended for performance here.
        # If not installed, trimesh falls back to other methods or raises errors depending on setup.

        # Optimization: Do nothing if the mesh is already small enough
        if len(mesh.faces) <= target_face_count:
            return (mesh,)

        # Perform decimation
        simplified_mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
        return (simplified_mesh,)


class SaveMeshSTL:
    """
    ComfyUI Node to save the mesh object as an .stl file.
    Saves to the configured ComfyUI output directory.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "The mesh object to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI_Mesh_",
                        "tooltip": "Prefix for the output filename. A 5-digit counter will be appended automatically.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True  # Tells ComfyUI that this node has side effects (saving files)
    CATEGORY = "DepthMap2STL"

    def save(self, mesh, filename_prefix):
        # Locate ComfyUI's output folder
        import os

        import folder_paths

        output_dir = folder_paths.get_output_directory()

        # Find the next available filename to avoid overwrites
        counter = 1
        while True:
            filename = f"{filename_prefix}{counter:05d}.stl"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        # Export using trimesh
        mesh.export(filepath)

        # Return UI feedback so the user knows where the file went
        return {"ui": {"status": [f"Saved to {filepath}"]}}


class PreviewMeshSTL:
    """
    ComfyUI Node to render a 3D isometric preview of the mesh.
    Returns an IMAGE that can be viewed in standard ComfyUI PreviewImage nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "The mesh object to preview."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "DepthMap2STL"

    def preview(self, mesh):
        from io import BytesIO

        import matplotlib.pyplot as plt

        # Setup pure-python rendering using Matplotlib
        # This avoids needing a heavy 3D engine just for a preview thumbnail.
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extract geometry
        v = mesh.vertices
        f = mesh.faces

        # Plot mesh surfaces
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2], triangles=f, cmap="viridis", edgecolor="none"
        )

        # Force equal aspect ratio for correct visualization dimensions
        # Handle cases where the mesh is flat (e.g. ptp=0) by replacing 0 with a small epsilon
        epsilon = 1e-6
        aspect = [np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2])]
        aspect = [a if a > epsilon else 1.0 for a in aspect]
        ax.set_box_aspect(aspect)

        # Hide chart axes
        plt.axis("off")

        # Render the plot to an in-memory buffer
        buf = BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True
        )
        buf.seek(0)
        img_pil = Image.open(buf).convert("RGB")
        plt.close(fig)

        # Convert PIL image to ComfyUI-compatible format
        # ComfyUI expects float32 [B, H, W, C] tensors in range 0-1
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_np = img_np[None, :, :, :]  # Add batch dimension

        # If 'torch' is available, return a Tensor (standard ComfyUI behavior).
        # If not (e.g., lightweight dev env), return numpy array (some nodes support this).
        try:
            import torch

            return (torch.from_numpy(img_np),)
        except ImportError:
            return (img_np,)


NODE_CLASS_MAPPINGS = {
    "DepthMapToMesh": DepthMapToMesh,
    "SimplifyMesh": SimplifyMesh,
    "SaveMeshSTL": SaveMeshSTL,
    "PreviewMeshSTL": PreviewMeshSTL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthMapToMesh": "Depth Map to Mesh",
    "SimplifyMesh": "Simplify Mesh",
    "SaveMeshSTL": "Save Mesh as STL",
    "PreviewMeshSTL": "Preview Mesh as Image",
}
