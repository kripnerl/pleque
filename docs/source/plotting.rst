.. _Plotting:

Plotting
========

PLEQUE ships two ready-made figures — the equilibrium *overview* plot and the *geometry*
(COCOS direction) plot — plus the individual layer routines they are built from, which live
in :mod:`pleque.utils.plotting` and can be used to compose a figure of your own.

All the examples below use the equilibria bundled with PLEQUE, so they can be reproduced
without any data files.

Overview plot
-------------

:meth:`~pleque.core.equilibrium.Equilibrium.plot_overview` draws the plasma cross section
with its first wall, separatrix, last closed flux surface, poloidal flux contours, the first
few scrape-off layer surfaces and the O-point/X-point markers.

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_overview()

Choosing the layers
-------------------

The overview plot is composed of layers. Each of them is controlled by a single argument
which accepts

* ``None`` — draw the layer with its default style (this is the default),
* ``False`` — do not draw the layer,
* a mapping of keyword arguments, merged over the layer defaults and passed on to the
  corresponding routine of :mod:`pleque.utils.plotting`.

=================== ================================================== =========================================
Argument            Routine                                            Layer
=================== ================================================== =========================================
``first_wall``      :func:`~pleque.utils.plotting.plot_first_wall`      First wall contour and its shaded interior
``separatrix``      :func:`~pleque.utils.plotting.plot_separatrix`      Separatrix, clipped to the first wall
``lcfs``            :func:`~pleque.utils.plotting.plot_lcfs`            Last closed flux surface
``psi_contours``    :func:`~pleque.utils.plotting.plot_psi_contours`    Poloidal flux contours inside the LCFS
``near_sol``        :func:`~pleque.utils.plotting.plot_near_sol`        First few flux surfaces in the SOL
``extremes``        :func:`~pleque.utils.plotting.plot_extremes`        Magnetic axis and X-point markers
=================== ================================================== =========================================

Turning layers off gives a bare plasma boundary:

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_overview(psi_contours=False, near_sol=False, extremes=False)

Restyling the layers
--------------------

A mapping keeps the layer but overrides its style. The accepted keys are those of the
corresponding routine and, beyond them, of the underlying matplotlib call:

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_overview(
       first_wall={"facecolor": "whitesmoke"},
       lcfs={"color": "k", "lw": 1},
       psi_contours={"levels": 40, "alpha": 0.3},
       near_sol={"dr": 5e-3, "colors": "C2"},
       extremes={"all_x_points": True},
   )

Selected flux contours
----------------------

The ``contours`` argument adds flux contours at explicitly chosen levels. Unlike the
``psi_contours`` layer, which is masked to the plasma, and unlike the separatrix, which is
clipped to the first wall, the selected contours are drawn over the whole computational grid
— that is what makes the separatrix branches of a double-null or snowflake configuration
visible.

Each contour is given as

* a number — used directly as :math:`\psi_N`, so ``1.0`` is the separatrix contour,
* a :class:`~pleque.core.coordinates.Coordinates` instance — the contour passing through the
  given point,
* the string ``"x_point"`` or ``"secondary_x_point"`` — the contour passing through the
  respective X-point of the equilibrium.

A single item or an iterable of them is accepted. Points the equilibrium does not have (the
secondary X-point of a single-null configuration, for instance) are skipped with a warning.

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_overview(contours=[1.0, "secondary_x_point"])

For a double-null equilibrium both X-points lie on the same flux surface, so the single
:math:`\psi_N = 1` contour already shows the whole topology:

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(2)
   eq.plot_overview(contours=1.0)

The selected contours are always drawn **first**, so that they lie underneath all the other
layers — where the standard layers are dense, the layers above them may have to be thinned
out for the selected contours to show. Their default colour and width come from the
``selected_contour_color`` and ``selected_contour_linewidth``
:doc:`configuration <configuration>` options; they can also be restyled per call by passing a
mapping with a ``levels`` key:

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_overview(contours={"levels": [0.5, 1.0], "colors": "C4", "linewidths": 2},
                    psi_contours={"alpha": 0.15}, near_sol=False)

Geometry (COCOS) plot
---------------------

:meth:`~pleque.core.equilibrium.Equilibrium.plot_geometry` shows the directions of the
toroidal and poloidal angles, of the magnetic field and of the plasma current, as implied by
the COCOS convention of the equilibrium (see :doc:`initialization`).

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_geometry()

Either view can be plotted alone, and the arrows, labels and layers can be restyled the same
way as in the overview plot:

.. plot::
   :include-source:

   from pleque.tests.utils import load_testing_equilibrium

   eq = load_testing_equilibrium(5)
   eq.plot_geometry(top_view=False, text_kwargs={"fontsize": 14},
                    arrow_kwargs={"color": "C3"})

Building your own figure
------------------------

Every layer routine takes an ``ax`` argument, so a figure can be assembled from scratch.
Routines which contour the poloidal flux also take a pre-computed ``coords`` grid, which is
worth passing when several of them are combined — otherwise each one evaluates
:math:`\psi` on its own grid.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt

   from pleque.tests.utils import load_testing_equilibrium
   from pleque.utils.plotting import (normalize_axis_xylim_by_first_wall, plot_first_wall,
                                      plot_lcfs, plot_rational_surface,
                                      plot_selected_contours)

   eq = load_testing_equilibrium(5)
   coords = eq.grid((400, 600), "size")

   fig, ax = plt.subplots(figsize=(4, 6))
   plot_first_wall(eq, ax, fill=False)
   plot_selected_contours(eq, ax, ["x_point", "secondary_x_point"], coords=coords,
                          colors="lightgrey")
   plot_lcfs(eq, ax, color="k", ls="-")
   plot_rational_surface(eq, ax, [(2, 1), (3, 2)], coords=coords)

   normalize_axis_xylim_by_first_wall(eq, ax)
   ax.set_aspect("equal")
   ax.set_xlabel("R [m]")
   ax.set_ylabel("Z [m]")

API reference
-------------

.. automodule:: pleque.utils.plotting
    :members:
    :undoc-members:
    :show-inheritance:
