import os
import sys
from unittest.mock import MagicMock, patch

# Set matplotlib backend to Agg to avoid GUI issues during tests
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from depth2mesh.nodes import (  # noqa: E402
    DepthMapToMesh,
    PreviewMeshSTL,
    SaveMeshSTL,
    SimplifyMesh,
)

# --- Fixtures and Mocks ---


class MockTensor:
    """Simulates a PyTorch tensor for ComfyUI inputs."""

    def __init__(self, numpy_array):
        self._numpy_array = numpy_array

    def __getitem__(self, idx):
        # Support indexing like tensor[0]
        if isinstance(idx, int):
            # Slicing the first dimension
            return MockTensor(self._numpy_array[idx])
        return MockTensor(self._numpy_array[idx])

    def cpu(self):
        return self

    def numpy(self):
        return self._numpy_array

    # Add dummy shape property if needed
    @property
    def shape(self):
        return self._numpy_array.shape


@pytest.fixture
def mock_image_tensor():
    # Batch=1, Height=10, Width=10, Channels=3
    # ComfyUI images are float 0-1
    data = np.random.rand(1, 10, 10, 3).astype(np.float32)
    return MockTensor(data)


@pytest.fixture
def mock_mesh():
    mesh = MagicMock()
    # Basic properties that might be accessed
    mesh.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh.faces = np.array([[0, 1, 2]])
    # Mock simplify method to return itself (or a copy)
    mesh.simplify_quadric_decimation.return_value = mesh
    return mesh


# --- Tests ---


def test_depth_map_to_mesh_generate(mock_image_tensor):
    # Arrange
    node = DepthMapToMesh()
    width = 10.0
    height = 10.0
    depth = 5.0
    power = 1.0

    # Act
    # Patch the core function to verify node logic without running expensive geometry
    with patch("depth2mesh.nodes.depth2mesh") as mock_core_func:
        mock_core_func.return_value = "MockTRIMESH"
        result = node.generate(mock_image_tensor, width, height, depth, power)

    # Assert
    assert result == ("MockTRIMESH",)
    mock_core_func.assert_called_once()

    # Verify the image passed to core was correct
    args, _ = mock_core_func.call_args
    img_arg = args[0]
    # Should be a PIL Image
    assert hasattr(img_arg, "size")


def test_simplify_mesh_performs_simplification(mock_mesh):
    # Arrange
    node = SimplifyMesh()
    # Set face count higher than target so it triggers simplification
    mock_mesh.faces = np.zeros((1000, 3))
    target = 500

    # Act
    result = node.simplify(mock_mesh, target)

    # Assert
    # Logic: simplified_mesh = mesh.simplify_quadric_decimation...
    mock_mesh.simplify_quadric_decimation.assert_called_with(face_count=target)
    assert result[0] == mock_mesh.simplify_quadric_decimation.return_value


def test_simplify_mesh_skips_if_already_small(mock_mesh):
    # Arrange
    node = SimplifyMesh()
    # Set face count lower than target
    mock_mesh.faces = np.zeros((10, 3))
    target = 500

    # Act
    result = node.simplify(mock_mesh, target)

    # Assert
    mock_mesh.simplify_quadric_decimation.assert_not_called()
    assert result[0] == mock_mesh


def test_save_mesh_stl(mock_mesh, tmp_path):
    # Arrange
    node = SaveMeshSTL()
    filename_prefix = "TEST_"

    # Mock folder_paths module which is ComfyUI specific
    mock_folder_paths = MagicMock()
    # Use tmp_path for output
    output_dir = str(tmp_path)
    mock_folder_paths.get_output_directory.return_value = output_dir

    with patch.dict(sys.modules, {"folder_paths": mock_folder_paths}):
        # Mock os.path.exists to simulate file conflict once
        # logic: while True: check exists.
        # We want:
        # Check TEST_00001.stl -> Exists
        # Check TEST_00002.stl -> Does not exist -> Break

        target_file_1 = os.path.join(output_dir, "TEST_00001.stl")
        target_file_2 = os.path.join(output_dir, "TEST_00002.stl")

        def exists_side_effect(path):
            if path == target_file_1:
                return True
            return False

        with patch("os.path.exists", side_effect=exists_side_effect):
            # Act
            result = node.save(mock_mesh, filename_prefix)

            # Assert
            mock_mesh.export.assert_called_with(target_file_2)
            assert "ui" in result
            assert "status" in result["ui"]


def test_preview_mesh_stl_returns_image(mock_mesh):
    # Arrange
    node = PreviewMeshSTL()

    # Act
    # This will use matplotlib to render
    result = node.preview(mock_mesh)

    # Assert
    assert len(result) == 1
    image_output = result[0]

    # Result should be either numpy array or torch tensor depending on environment of runner
    # We check for general shape compatibility [B, H, W, C]
    if hasattr(image_output, "shape"):
        assert len(image_output.shape) == 4
        assert image_output.shape[0] == 1  # Batch size 1
        assert image_output.shape[3] == 3  # RGB channels
