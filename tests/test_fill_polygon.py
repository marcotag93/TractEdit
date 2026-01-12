import numpy as np
import unittest


def fill_polygon_logic(roi_data, vox_points_float, shape, view_type):
    if len(vox_points_float) < 3:
        return False

    const_axis = 2
    plane_axes = (0, 1)

    if view_type == "axial":
        const_axis = 2
        plane_axes = (0, 1)
    elif view_type == "coronal":
        const_axis = 1
        plane_axes = (0, 2)
    elif view_type == "sagittal":
        const_axis = 0
        plane_axes = (1, 2)

    const_val = int(np.round(np.mean(vox_points_float[:, const_axis])))
    if const_val < 0 or const_val >= shape[const_axis]:
        return False

    poly = vox_points_float[:, plane_axes]

    min_x = int(np.floor(np.min(poly[:, 0])))
    max_x = int(np.ceil(np.max(poly[:, 0])))
    min_y = int(np.floor(np.min(poly[:, 1])))
    max_y = int(np.ceil(np.max(poly[:, 1])))

    min_x = max(0, min_x)
    max_x = min(shape[plane_axes[0]] - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(shape[plane_axes[1]] - 1, max_y)

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)

    if len(x_range) == 0 or len(y_range) == 0:
        return False

    xx, yy = np.meshgrid(x_range, y_range)
    points = np.vstack((xx.flatten(), yy.flatten())).T

    if len(points) == 0:
        return False

    path = poly
    n_vertices = len(path)
    j = n_vertices - 1

    x = points[:, 0]
    y = points[:, 1]

    mask = np.zeros(len(points), dtype=bool)

    for i in range(n_vertices):
        xi, yi = path[i]
        xj, yj = path[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
        )
        mask ^= intersect
        j = i

    mask_grid = mask.reshape(len(y_range), len(x_range))
    mask_transposed = mask_grid.T

    slices = [slice(None)] * 3
    slices[const_axis] = const_val
    slices[plane_axes[0]] = slice(min_x, max_x + 1)
    slices[plane_axes[1]] = slice(min_y, max_y + 1)

    roi_patch = roi_data[tuple(slices)]
    roi_patch[mask_transposed] = 1

    return True


class TestPolygonFill(unittest.TestCase):
    def test_square_fill_axial(self):
        shape = (10, 10, 10)
        roi_data = np.zeros(shape, dtype=np.uint8)

        # Draw a square in slice Z=5
        # (2,2) to (7,2) to (7,7) to (2,7)
        points = np.array(
            [[2.0, 2.0, 5.0], [7.0, 2.0, 5.0], [7.0, 7.0, 5.0], [2.0, 7.0, 5.0]]
        )

        fill_polygon_logic(roi_data, points, shape, "axial")

        # Check center is filled
        self.assertEqual(roi_data[5, 5, 5], 1)
        # Check outside
        self.assertEqual(roi_data[1, 1, 5], 0)
        self.assertEqual(roi_data[8, 8, 5], 0)
        # Check corners
        self.assertEqual(
            roi_data[2, 2, 5], 1
        )  # Boundary might vary slightly depending on rounding, but inside should be 1

    def test_triangle_fill_coronal(self):
        shape = (10, 10, 10)
        roi_data = np.zeros(shape, dtype=np.uint8)

        # Draw triangle in slice Y=5
        # Plane is (X, Z) -> (0, 2)
        points = np.array([[2.0, 5.0, 2.0], [7.0, 5.0, 2.0], [4.5, 5.0, 7.0]])

        fill_polygon_logic(roi_data, points, shape, "coronal")

        self.assertEqual(roi_data[4, 5, 4], 1)
        self.assertEqual(roi_data[2, 5, 2], 1)  # Vertex


if __name__ == "__main__":
    unittest.main()
