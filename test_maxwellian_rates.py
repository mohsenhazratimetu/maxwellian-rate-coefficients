import unittest

import numpy as np
from scipy.constants import e, m_e

from maxwellian_rates import rate_coefficients


class MaxwellianRateTests(unittest.TestCase):
    def test_constant_cross_section_across_a_wide_grid(self):
        temperatures = np.array([1e-60, 0.27, 1.0, 100.0])
        energy = [0.0, 0.1, 1e4, 1e6]
        sigma = 1e-20
        expected = sigma * np.sqrt(8 * e * temperatures / (np.pi * m_e))
        actual = rate_coefficients(energy, [sigma] * len(energy), temperatures)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

    def test_threshold_and_finite_upper_limit(self):
        energy = np.array([2.0, 7.0])
        sigma = 2e-20
        temperatures = np.array([0.1, 2.0, 15.0])
        lower = energy[0] / temperatures
        upper = energy[1] / temperatures
        expected = sigma * np.sqrt(8 * e * temperatures / (np.pi * m_e))
        expected *= (1 + lower) * np.exp(-lower) - (1 + upper) * np.exp(-upper)
        actual = rate_coefficients(energy, [sigma, sigma], temperatures)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

    def test_linear_cross_section_near_zero(self):
        energy = [0.0, 1e6]
        temperatures = np.array([1e-60, 1.0, 100.0])
        slope = 1e-26
        expected = 2 * slope * temperatures * np.sqrt(8 * e * temperatures / (np.pi * m_e))
        actual = rate_coefficients(energy, [0.0, slope * energy[-1]], temperatures)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
