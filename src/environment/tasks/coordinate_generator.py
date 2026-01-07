"""
Coordinate generation for intra-room task positioning.

Generates realistic pickup/delivery coordinates within node boundaries.
"""
import numpy as np
from typing import Tuple, Optional


def generate_intra_room_coordinates(
    node_center_x: float,
    node_center_y: float,
    node_width: float,
    node_height: float,
    margin: float = 0.5,
    num_points: int = 1
) -> list:
    """
    Generate random coordinates within a node's boundaries.

    Args:
        node_center_x: Node center X coordinate
        node_center_y: Node center Y coordinate
        node_width: Node width (meters)
        node_height: Node height (meters)
        margin: Safety margin from walls (meters)
        num_points: Number of coordinate pairs to generate

    Returns:
        List of (x, y) tuples within the node boundaries
    """
    # Calculate usable area (minus margin)
    usable_width = max(node_width - 2 * margin, 0.5)
    usable_height = max(node_height - 2 * margin, 0.5)

    # Calculate boundaries
    x_min = node_center_x - usable_width / 2
    x_max = node_center_x + usable_width / 2
    y_min = node_center_y - usable_height / 2
    y_max = node_center_y + usable_height / 2

    # Generate random points
    coordinates = []
    for _ in range(num_points):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        coordinates.append((x, y))

    return coordinates


def generate_shelf_coordinates(
    node_center_x: float,
    node_center_y: float,
    node_width: float,
    node_height: float,
    num_shelves: int = 4,
    margin: float = 0.5
) -> list:
    """
    Generate coordinates along walls (simulating shelf locations).

    Useful for storage nodes where items are stored on shelves.

    Args:
        node_center_x: Node center X
        node_center_y: Node center Y
        node_width: Node width
        node_height: Node height
        num_shelves: Number of shelf positions
        margin: Distance from center (closer to walls)

    Returns:
        List of (x, y) tuples along node perimeter
    """
    coordinates = []

    # Shelf positions along walls
    half_w = (node_width / 2) - margin
    half_h = (node_height / 2) - margin

    shelves_per_wall = num_shelves // 4

    # North wall
    for i in range(shelves_per_wall):
        x = node_center_x + np.random.uniform(-half_w, half_w)
        y = node_center_y + half_h
        coordinates.append((x, y))

    # South wall
    for i in range(shelves_per_wall):
        x = node_center_x + np.random.uniform(-half_w, half_w)
        y = node_center_y - half_h
        coordinates.append((x, y))

    # East wall
    for i in range(shelves_per_wall):
        x = node_center_x + half_w
        y = node_center_y + np.random.uniform(-half_h, half_h)
        coordinates.append((x, y))

    # West wall
    for i in range(shelves_per_wall):
        x = node_center_x - half_w
        y = node_center_y + np.random.uniform(-half_h, half_h)
        coordinates.append((x, y))

    return coordinates


def generate_bed_coordinates(
    node_center_x: float,
    node_center_y: float,
    node_width: float,
    node_height: float,
    num_beds: int = 2
) -> list:
    """
    Generate coordinates for hospital beds in a recovery ward.

    Beds are typically arranged along walls with space in the middle.

    Args:
        node_center_x: Node center X
        node_center_y: Node center Y
        node_width: Node width
        node_height: Node height
        num_beds: Number of beds in room

    Returns:
        List of (x, y) tuples at bed locations
    """
    coordinates = []

    half_w = node_width / 2
    half_h = node_height / 2

    if num_beds == 1:
        # Single bed in center
        coordinates.append((node_center_x, node_center_y))

    elif num_beds == 2:
        # Two beds on opposite walls
        bed_offset = half_w * 0.6
        coordinates.append((node_center_x - bed_offset, node_center_y))
        coordinates.append((node_center_x + bed_offset, node_center_y))

    elif num_beds == 4:
        # Four beds (2x2 grid)
        x_offset = half_w * 0.5
        y_offset = half_h * 0.5
        coordinates.append((node_center_x - x_offset, node_center_y - y_offset))
        coordinates.append((node_center_x + x_offset, node_center_y - y_offset))
        coordinates.append((node_center_x - x_offset, node_center_y + y_offset))
        coordinates.append((node_center_x + x_offset, node_center_y + y_offset))

    else:
        # Distribute evenly along perimeter
        angle_step = 2 * np.pi / num_beds
        radius_x = half_w * 0.7
        radius_y = half_h * 0.7

        for i in range(num_beds):
            angle = i * angle_step
            x = node_center_x + radius_x * np.cos(angle)
            y = node_center_y + radius_y * np.sin(angle)
            coordinates.append((x, y))

    return coordinates


def sample_task_coordinates(
    from_node,
    to_node,
    task_type: str,
    enable_intra_room: bool = True,
    same_node_offset: float = 1.0
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Generate pickup and delivery coordinates for a task.

    Args:
        from_node: Source HospitalNode
        to_node: Destination HospitalNode
        task_type: Type of task ('replenishment', 'returns', etc.)
        enable_intra_room: If True, generate coordinates within rooms
        same_node_offset: If task is within same node, offset between pickup/delivery

    Returns:
        Tuple of (from_coordinates, to_coordinates)
        Returns (None, None) if intra_room disabled or nodes have no dimensions
    """
    if not enable_intra_room:
        return None, None

    # Check if nodes have dimensions
    if not hasattr(from_node, 'width') or not hasattr(to_node, 'width'):
        return None, None

    # Generate pickup coordinates
    if task_type == 'replenishment' and from_node.node_type == 'storage':
        # Pickup from shelf in storage
        from_coords_list = generate_shelf_coordinates(
            from_node.center_x,
            from_node.center_y,
            from_node.width,
            from_node.height,
            num_shelves=8
        )
        from_coords = from_coords_list[np.random.randint(len(from_coords_list))]
    else:
        # Random position in room
        from_coords = generate_intra_room_coordinates(
            from_node.center_x,
            from_node.center_y,
            from_node.width,
            from_node.height,
            num_points=1
        )[0]

    # Generate delivery coordinates
    if to_node.node_type == 'recovery':
        # Deliver to bed/patient location
        bed_coords_list = generate_bed_coordinates(
            to_node.center_x,
            to_node.center_y,
            to_node.width,
            to_node.height,
            num_beds=2  # Assume 2-bed rooms
        )
        to_coords = bed_coords_list[np.random.randint(len(bed_coords_list))]

    elif to_node.node_type == 'storage':
        # Deliver to shelf
        shelf_coords_list = generate_shelf_coordinates(
            to_node.center_x,
            to_node.center_y,
            to_node.width,
            to_node.height,
            num_shelves=8
        )
        to_coords = shelf_coords_list[np.random.randint(len(shelf_coords_list))]

    else:
        # Random position
        to_coords = generate_intra_room_coordinates(
            to_node.center_x,
            to_node.center_y,
            to_node.width,
            to_node.height,
            num_points=1
        )[0]

    # If task is within same node, ensure pickup and delivery are different
    if from_node.node_id == to_node.node_id:
        # Offset delivery point from pickup
        to_coords = (
            to_coords[0] + np.random.uniform(-same_node_offset, same_node_offset),
            to_coords[1] + np.random.uniform(-same_node_offset, same_node_offset)
        )

        # Clamp to node boundaries
        x_min, y_min, x_max, y_max = to_node.bounds
        to_coords = (
            np.clip(to_coords[0], x_min + 0.5, x_max - 0.5),
            np.clip(to_coords[1], y_min + 0.5, y_max - 0.5)
        )

    return from_coords, to_coords


def visualize_coordinates(node, coordinates: list):
    """
    Print ASCII visualization of coordinates within a node (for debugging).

    Args:
        node: HospitalNode
        coordinates: List of (x, y) tuples to visualize
    """
    print(f"\n{node.node_id} ({node.width}m × {node.height}m)")
    print(f"Center: ({node.center_x:.1f}, {node.center_y:.1f})")
    print("\nCoordinates:")

    x_min, y_min, x_max, y_max = node.bounds

    # Create simple ASCII grid
    grid_size = 20
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

    # Draw boundaries
    for i in range(grid_size):
        grid[0][i] = '-'
        grid[grid_size-1][i] = '-'
        grid[i][0] = '|'
        grid[i][grid_size-1] = '|'

    # Plot coordinates
    for idx, (x, y) in enumerate(coordinates):
        # Normalize to grid
        grid_x = int(((x - x_min) / (x_max - x_min)) * (grid_size - 2)) + 1
        grid_y = int(((y - y_min) / (y_max - y_min)) * (grid_size - 2)) + 1

        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            grid[grid_y][grid_x] = str(idx % 10)

        print(f"  {idx}: ({x:.2f}, {y:.2f})")

    # Print grid
    print("\nVisualization:")
    for row in grid:
        print(''.join(row))


# Example usage
if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class MockNode:
        node_id: str
        node_type: str
        center_x: float
        center_y: float
        width: float
        height: float

        @property
        def bounds(self):
            half_w = self.width / 2
            half_h = self.height / 2
            return (
                self.center_x - half_w,
                self.center_y - half_h,
                self.center_x + half_w,
                self.center_y + half_h
            )

    # Test with storage node
    storage = MockNode('Storage_A', 'storage', 5.0, 5.0, 4.0, 4.0)
    shelf_coords = generate_shelf_coordinates(
        storage.center_x, storage.center_y,
        storage.width, storage.height,
        num_shelves=8
    )
    visualize_coordinates(storage, shelf_coords)

    # Test with recovery ward
    ward = MockNode('Ward_10817', 'recovery', 20.0, 10.0, 5.0, 4.0)
    bed_coords = generate_bed_coordinates(
        ward.center_x, ward.center_y,
        ward.width, ward.height,
        num_beds=2
    )
    visualize_coordinates(ward, bed_coords)

    # Test task coordinate generation
    print("\n" + "="*50)
    print("Task: Replenishment from Storage → Ward")
    from_coords, to_coords = sample_task_coordinates(
        storage, ward, 'replenishment', enable_intra_room=True
    )
    print(f"Pickup at: {from_coords}")
    print(f"Delivery at: {to_coords}")
