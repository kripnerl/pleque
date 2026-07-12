Equilibrium initialisation
==========================

The central object of PLEQUE is the :class:`pleque.Equilibrium` class. It is
usually constructed by one of the readers in :mod:`pleque.io` (e.g.
:func:`pleque.io.readers.read_geqdsk`), which build an ``xarray.Dataset`` with
the poloidal flux function :math:`\psi(R, Z)` on a rectangular grid and 1D
profiles of the toroidal field function :math:`F` and pressure :math:`p` (or
their :math:`\psi`-derivatives ``FFprime`` and ``pprime``) as functions of
:math:`\psi_\mathrm{N}`.

Initialisation sequence
-----------------------

#. **Input parsing.** Spatial data, profiles and optional hints are read from
   the dataset and constructor arguments. If no first wall is provided, an
   artificial rectangular wall slightly inset with respect to the
   :math:`\psi`-grid is synthesized (``grid.synthetic_wall_margin`` and
   ``grid.synthetic_wall_points_per_side`` settings).
#. **2D spline construction.** :math:`\psi(R, Z)` is interpolated by a
   bivariate spline (``splines.psi_order`` / ``splines.psi_smooth`` settings).
#. **Critical points.** Candidate extremes of :math:`|\nabla\psi|^2` are
   located on a regular grid and classified by the determinant of the Hessian
   of :math:`\psi` into O-points (local extrema) and X-points (saddle points).
   The magnetic axis and the relevant X-points are recognized among the
   candidates, and the plasma configuration (limiter vs. diverted) is
   determined together with the limiter point and :math:`\psi` on the last
   closed flux surface.
#. **Plasma boundary.** Strike points are found as intersections of the
   LCFS-:math:`\psi` contour with the first wall. The LCFS contour is located
   by the marching-squares algorithm on a search grid and refined by an
   iterative downhill (gradient-step) method until the relative :math:`\psi`
   error drops below ``lcfs.refinement_tolerance``; for diverted plasmas the
   X-point is inserted into the contour. Poloidal field-line tracing of the
   boundary is *not* used during the initialisation; it is available
   separately via :meth:`pleque.Equilibrium.lcfs_field_line`.
#. **1D profile splines.** If derivatives (``pprime``, ``FFprime``) are
   provided — the preferred input, as it avoids differentiating noisy data —
   the :math:`p` and :math:`F` profiles are obtained by integration (this
   requires :math:`\psi_\mathrm{axis}` and :math:`\psi_\mathrm{LCFS}`, which
   is why this step runs after the critical-point search). Otherwise the
   derivatives are computed from the provided profiles. If neither pressure
   nor :math:`F` is available, a vacuum equilibrium (:math:`p = F = 0`) is
   constructed.
#. **Midplane mapping.** A 1D spline mapping the outer-midplane radius to
   :math:`\psi` is built.

Hints and ``init_method``
-------------------------

The optional arguments ``mg_axis``, ``psi_lcfs``, ``x_points`` and
``strike_points`` act as *hints*. Each may alternatively be provided as a
dataset variable of the same name; an explicitly passed argument takes
precedence (with a logged warning). The ``init_method`` argument controls how
the hints are used:

``"full"``
    All hints are ignored; the module recognizes all critical points itself.
``"hints"`` (default)
    The hints assist the recognition of the magnetic axis and X-points.
``"fast_forward"``
    The ``mg_axis`` and ``x_points`` hints are taken as final and the
    critical-point search is skipped; ``psi_lcfs`` and ``strike_points``
    hints are likewise used as final when given. If the required hints are
    missing, the initialisation falls back to ``"hints"`` with a logged
    warning.

Diagnostics
-----------

PLEQUE logs through the standard :mod:`logging` machinery under the
``pleque`` logger; see :ref:`logging` for how to enable it. If the
initialisation fails, the exception is logged and re-raised; enabling the
``debug_plots`` setting (e.g. ``PLEQUE_DEBUG_PLOTS=1``) additionally draws a
debug plot of the partially initialised equilibrium.
