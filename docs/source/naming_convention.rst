Naming convention used in PLEQUE
================================

Coordinates
-----------


Here presented naming convention is used to read/create input/output dict/xarray files.

* 2D
    * ``R`` (default): Radial cylindrical coordinates with zero on machine axis
    * ``Z`` (default): Vertical coordinate with zero on machine geometrical axis
* 1D
    * ``psi_n`` (default): Normalized poloidal magnetic flux with zero on magnetic axis and one
      on the last closed flux surface

      :math:`\psi_\mathrm{N} = \frac{\psi -
      \psi_\text{ax}}{\psi_\text{LCFS} - \psi_\text{ax}}`
    * Fallowing input options are not implemented yet.
    * ``rho``: :math:`\rho = \sqrt{\psi_\text{N}}`
    * ``psi_1dprof`` - poloidal magnetic flux; this coordinate axis is used only if ``psi_n`` is
      not found on the input. Output files uses implicitly ``psi_n`` axis.

2D profiles
-----------
* Required on the input
    * ``psi`` (Wb): poloidal magnetic flux
* Calculated
    * ``B_R`` (T): :math:`R` component of the magnetic field.
    * ``B_Z`` (T): :math:`Z` component of the magnetic field.
    * ``B_pol`` (T): Poloidal component of the magnetic field. :math:`B_\theta =\text{sign\,} (I_p) \sqrt{B_R^2 + B_Z^2}`
      **Todo** resolve the sign of B_pol and implement it!!!
    * ``B_tor`` (T): Toroidal component of the magnetic field.
    * ``B_abs`` (T): Absolute value of the magnetic field.
    * ``j_R`` (A/m2): :math:`R` component of the current density.
      **todo: Check the current unit**
    * ``j_Z`` (A/m2): :math:`Z` component of the current density.
    * ``j_pol`` (A/m2): Poloidal component of the current density.
    * ``j_tor`` (A/m2): Toroidal component of the current density.
    * ``j_abs`` (A/m2): Asolute value of the current density.

1D profiles
-----------

* Required on the input
    * ``pressure`` (Pa)
    * ``pprime`` (Pa/Wb)
    * ``F``: :math:`F = R B_\phi`

* Calculated
    * ``pprime``: :math:`p \partial_\psi`
    * ``Fprime``: :math:`F' = \partial_\psi F`
    * ``FFprime``: :math:`FF' = F \partial_\psi F`
    * ``fprime``: :math:`f' = \partial_\psi f`
    * ``f``: :math:`f = (1/\mu_0) R B_\phi`
    * ``ffprime``: :math:`ff' = f \partial_\psi f`
    * ``rho``, ``psi_n``

* Deriver
    * `q`: safety factor profile
    * `qprime`:math:`q' = \partial_\psi q`
    * Not yet implemented:
        * `magnetic_shear`
        * ...


Attributes
----------
* To be written.

FluxSurface quantities
----------------------

