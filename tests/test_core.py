import numpy as np
import pytest
import trimesh
from PIL import Image

from depth2mesh.core import depth2mesh


def create_synthetic_image(width, height, shape="square"):
    """Creates a synthetic RGBA image for testing."""
    # Create RGBA image
    data = np.zeros((height, width, 4), dtype=np.uint8)
    # Full opacity alpha
    data[:, :, 3] = 255
    # some gradient in RGB
    for y in range(height):
        for x in range(width):
            data[y, x, :3] = int((x + y) * 10) % 255

    if shape == "circle":
        # Mask out corners
        cy, cx = height // 2, width // 2
        r = min(width, height) // 2
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
        data[~mask, 3] = 0

    return Image.fromarray(data)


def test_depth2mesh_basic_square():
    # Arrange
    img = create_synthetic_image(10, 10, shape="square")
    width_mm = 10.0
    height_mm = 10.0
    depth_mm = 5.0

    # Act
    mesh = depth2mesh(img, width_mm, height_mm, depth_mm)

    # Assert
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_depth2mesh_with_transparency_validation():
    # Arrange
    # Create image with transparent pixels (circle in 20x20)
    img = create_synthetic_image(20, 20, shape="circle")

    # Act
    mesh = depth2mesh(img, 20, 20, 5)

    # Assert
    assert isinstance(mesh, trimesh.Trimesh)
    # Note: depends on networkx for repair if not watertight by default
    assert mesh.is_watertight

    # Check bounds roughly
    bounds = mesh.bounds
    # z min should be 0 (bottom)
    # Use isclose for float comparison
    assert np.isclose(bounds[0][2], 0.0, atol=1e-5)
    # z max should be <= 5.0
    assert bounds[1][2] <= 5.0 + 1e-5


def test_depth2mesh_power_transformation():
    """Verifies that the power parameter affects the height distribution."""
    # Create a 10x10 gradient image
    data = np.zeros((10, 10, 4), dtype=np.uint8)
    data[:, :, 3] = 255
    for i in range(10):
        data[:, i, :3] = int((i / 9.0) * 255)
    img = Image.fromarray(data)

    # Act
    mesh_linear = depth2mesh(img, 10, 10, 10, power=1.0)
    mesh_power = depth2mesh(img, 10, 10, 10, power=2.0)

    # Assert
    # Power > 1 should result in lower average height for a linear gradient [0,1]^2 < [0,1]
    assert np.mean(mesh_power.vertices[:, 2]) < np.mean(mesh_linear.vertices[:, 2])


def test_depth2mesh_small_image():
    """Tests that the function handles a minimal 2x2 image."""
    img = Image.new("RGBA", (2, 2), (255, 255, 255, 255))
    mesh = depth2mesh(img, 10, 10, 5)
    assert len(mesh.vertices) > 0
    assert mesh.is_watertight


def test_depth2mesh_all_transparent_raises_error():
    # Arrange
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))

    # Act & Assert
    with pytest.raises(ValueError, match="All pixels are transparent"):
        depth2mesh(img, 10, 10, 10)


def test_depth2mesh_zero_height_raises_error():
    # Arrange
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 255))  # Black opaque -> 0 height

    # Act & Assert
    with pytest.raises(ValueError, match="Maximum height value is zero"):
        depth2mesh(img, 10, 10, 10)


def test_depth2mesh_input_path(tmp_path):
    # Arrange
    img = create_synthetic_image(10, 10, shape="square")
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    # Act
    mesh = depth2mesh(str(img_path), 10, 10, 5)

    # Assert
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight


def test_depth2mesh_donut_shape():
    """Tests an image with an internal hole (alpha=0 in the middle)."""
    # Create 20x20 image, opaque square with 6x6 transparent hole in middle
    data = np.zeros((20, 20, 4), dtype=np.uint8)
    data[:, :, :3] = 200  # grey
    data[:, :, 3] = 255  # opaque

    # Cut hole
    data[7:13, 7:13, 3] = 0

    img = Image.fromarray(data)

    # Act
    mesh = depth2mesh(img, 20, 20, 5)

    # Assert
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight
    # Should have internal walls now
    assert len(mesh.faces) > 500  # Roughly
