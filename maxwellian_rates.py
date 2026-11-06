"""Compute Maxwellian electron-impact rate coefficients from tabulated data."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.constants import e, m_e
from scipy.special import roots_legendre


_SPEED_FACTOR = np.sqrt(8 * e / (np.pi * m_e))
_NODES, _WEIGHTS = roots_legendre(32)
_UNIT_NODES = (_NODES + 1) / 2
_UNIT_WEIGHTS = _WEIGHTS / 2


def rate_coefficients(energy_ev, cross_section_m2, temperature_ev):
    """Return rates in m^3/s for electron temperatures in eV."""
    energy = np.asarray(energy_ev, dtype=float)
    sigma = np.asarray(cross_section_m2, dtype=float)
    temperatures = np.atleast_1d(np.asarray(temperature_ev, dtype=float))

    if energy.ndim != 1 or energy.size < 2 or sigma.shape != energy.shape:
        raise ValueError("Supply at least two matching energy and cross-section values.")
    if not np.all(np.isfinite(energy)) or energy[0] < 0 or np.any(np.diff(energy) <= 0):
        raise ValueError("Energies must be finite, nonnegative and strictly increasing.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("Cross sections must be finite and nonnegative.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("Supply a scalar or a one-dimensional temperature array.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("Temperatures must be finite and positive.")

    energy_width = np.diff(energy)
    rates = np.empty(temperatures.shape)
    for index, temperature in enumerate(temperatures):
        # Each interval uses x = E / Te and y = x - x_left.
        with np.errstate(over="ignore"):
            left = energy[:-1] / temperature
            width = energy_width / temperature
        active = np.isfinite(left) & (width > 0)
        left = left[active]
        width = width[active]

        # Within an interval the integrand is a quadratic times exp(-y).
        # Beyond y = 50 the omitted fraction is below double precision.
        span = np.minimum(width, 50.0)
        y = span[:, None] * _UNIT_NODES
        fraction = y / width[:, None]
        sigma_left = sigma[:-1][active, None]
        sigma_right = sigma[1:][active, None]
        interpolated = sigma_left * (1 - fraction) + sigma_right * fraction

        x = left[:, None] + y
        integral = np.dot(span, (x * np.exp(-x) * interpolated) @ _UNIT_WEIGHTS)
        rates[index] = _SPEED_FACTOR * np.sqrt(temperature) * integral
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cross_sections", type=Path, help="Two columns: energy in eV and cross section in m^2")
    parser.add_argument("-o", type=Path, help="Output TSV path. The default is standard output.")
    parser.add_argument("-t", type=float, nargs=2, metavar=("MIN", "MAX"), default=(1e-60, 100.0),
                        help="Electron temperature limits in eV")
    parser.add_argument("-n", type=int, default=400, help="Number of log-spaced temperatures")
    args = parser.parse_args()

    low, high = args.t
    if not (0 < low < high < np.inf) or args.n < 2:
        parser.error("Use finite temperature limits with 0 < MIN < MAX and at least two points.")
    try:
        data = np.loadtxt(args.cross_sections, comments=("#", "%"), ndmin=2)
        if data.shape[1] != 2:
            parser.error("The input file must contain exactly two columns.")
        temperatures = np.geomspace(low, high, args.n)
        rates = rate_coefficients(data[:, 0], data[:, 1], temperatures)
        target = args.o if args.o is not None else sys.stdout
        np.savetxt(target, np.column_stack((temperatures, rates)), fmt="%.12e",
                   delimiter="\t", header="Te_eV\tk_m3_per_s", comments="% ")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
