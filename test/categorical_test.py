import unittest

import numpy as np

from distributions.categorical import Categorical


class TestCategorical(unittest.TestCase):
    def test_mle(self):
        # Test 1D input
        data = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        cat = Categorical.mle(data)
        expected_p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_almost_equal(cat.p, expected_p)

        # Test 2D input
        data = np.array([[0], [0], [1], [1], [2], [2], [3], [3], [4]])
        cat = Categorical.mle(data)
        expected_p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_almost_equal(cat.p, expected_p)

        # Test Laplace smoothing
        data = np.array([0, 0, 1, 1, 2])
        cat = Categorical.mle(data, alpha=1.0)
        expected_p = np.array([0.3, 0.3, 0.4])
        np.testing.assert_almost_equal(cat.p, expected_p)

        # Test invalid input
        with self.assertRaises(ValueError):
            Categorical.mle(np.array([[0, 1], [1, 2]]))


if __name__ == '__main__':
    unittest.main()
