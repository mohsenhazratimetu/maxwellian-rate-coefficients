# Maxwellian electron-impact rate coefficients

A Python tool to calculate $\langle \sigma v \rangle$ from a table of electron-impact cross sections. It writes a two-column table for use in plasma-chemistry calculations and COMSOL interpolation functions.

## Run the example

```sh
python -m pip install -r requirements.txt
python maxwellian_rates.py sample_cross_sections.tsv -t 0.01 100 -n 400 -o rates.tsv
python -m unittest -v
```

The input file has two whitespace-separated columns. Give electron energy in eV in ascending order and cross section in m². The program interpolates linearly between data points and sets the cross section to zero outside the table. To represent a reaction threshold, start the table at the threshold or include zero cross sections up to it. The example table contains synthetic data for a numerical check.

The output contains electron temperature $T_e$ in eV and rate coefficient $k$ in m³/s. In a COMSOL interpolation function, set the argument unit to eV and the function unit to m³/s. If the model uses mean electron energy $\bar{E}$ as its variable, evaluate the table at $T_e = 2\bar{E}/3$ for a Maxwellian distribution.

## Calculation

For electron energy $E$ and electron temperature $T_e$ in eV,

$$
k(T_e)=\sqrt{\frac{8e}{\pi m_e}}\,T_e^{-3/2}
\int_0^\infty E\,\sigma(E)\,\exp(-E/T_e)\,\mathrm{d}E .
$$

Here $e$ is the elementary charge and $m_e$ is the electron mass. The program integrates on each interpolation interval in $x=E/T_e$. A 32-point Gauss–Legendre rule resolves the weighted integrand near the start of each interval. The contribution more than 50 temperature scales beyond that start is below double precision for the piecewise linear cross section.

For the example cross section of $10^{-20}$ m² and $T_e=1$ eV, the analytical rate is $6.69238 \times 10^{-15}$ m³/s. The tests check this wide energy grid, a finite threshold and a cross section that rises linearly from zero.

The calculation assumes a Maxwellian electron energy distribution. [Rate coefficient reference](https://crane-plasma-chemistry.readthedocs.io/en/latest/rates/RateCoeff_Maxwellian.html)
