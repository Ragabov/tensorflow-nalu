import unittest

from experiments.utils import *

class UtilsTest(unittest.TestCase):

    def test_add_generate_synthetic_arithmetic_dataset(self):
        X, Y, boundaries = generate_synthetic_arithmetic_dataset("add", 0, 100, 100, 2)
        print(X.shape)
        expected_output = np.array([np.sum(X[i][boundaries[0]:boundaries[1]]) +
                                    np.sum(X[i][boundaries[2]:boundaries[3]]) for i in range(2)])

        np.testing.assert_allclose(Y, expected_output)

    def test_mult_generate_synthetic_arithmetic_dataset(self):
        X, Y, boundaries = generate_synthetic_arithmetic_dataset("mult", 0, 100, 100, 2)
        print(X.shape)
        expected_output = np.array([np.sum(X[i][boundaries[0]:boundaries[1]]) *
                                    np.sum(X[i][boundaries[2]:boundaries[3]]) for i in range(2)])

        np.testing.assert_allclose(Y, expected_output)

