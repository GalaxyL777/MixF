MixF Code is A Stokes Parameter Fitting Program Based on Analytical Solutions of Propagation Equations

The MixF code is a program designed to fit Stokes parameters using analytical solutions of radiative transfer equations (Wang et al. 2025). It supports two plasma regimes for users to choose from:
1. Cold Plasma – Applicable when γ=kT/mec2≪1γ=kT/mec2≪1.
2. Hot Plasma – Applicable when γ≫1γ≫1.
Note: Parameter X should not be excessively large.

Key Features
Accounts for absorption, Faraday rotation, and Faraday conversion in the plasma medium.

We also provided a simplified version of MixF, which is available when absorption can be neglected (i.e., when total polarization remain frequency-independent). This version retains only the mixing Faraday term, simplifying the fitting process.
Since the distance to the plasma medium is typically unknown, we recommend including both a background and a foreground RM layer.

Pre-Fitting Considerations
Before running MixF, users should estimate the ratio of Faraday rotation to Faraday conversion coefficients. If Faraday rotation dominates significantly, MCMC fitting—considering observational errors—may converge to an unphysical magnetic field strength.
To mitigate this, we recommend using the RM & CM (Rotation Measure & Conversion Measure) fitting version. Although this version does not directly output magnetic field strength or column density, users can still estimate these quantities using the definitions of RM and CM.



If you have any further questions or comments, feel free to contact to Weiyang (wywang@ucas.ac.cn) or Xiaohui (liuxh@bao.ac.cn)
