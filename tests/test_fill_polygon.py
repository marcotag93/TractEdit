import unittest

import numpy as np

from tractedit_pkg.visualization.drawing import DrawingManager


class TestPolygonFill(unittest.TestCase):
    def test_square_fill_axial(self):
        shape = (10, 10, 10)
        roi_data = np.zeros(shape, dtype=np.uint8)

        # Draw a square in slice Z=5
        # (2,2) to (7,2) to (7,7) to (2,7)
        points = np.array(
            [[2.0, 2.0, 5.0], [7.0, 2.0, 5.0], [7.0, 7.0, 5.0], [2.0, 7.0, 5.0]]
        )

        self.assertTrue(
            DrawingManager._fill_polygon(None, "roi", roi_data, points, shape, "axial")
        )

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

        self.assertTrue(
            DrawingManager._fill_polygon(None, "roi", roi_data, points, shape, "coronal")
        )

        self.assertEqual(roi_data[4, 5, 4], 1)
        self.assertEqual(roi_data[2, 5, 2], 1)  # Vertex


if __name__ == "__main__":
    unittest.main()
