Configuration
=============

PLEQUE's algorithms are controlled by a set of tunable parameters — grid
resolutions, solver tolerances, search heuristics, plotting defaults. All of
them have sensible built-in defaults, and all of them can be overridden at
three levels:

1. **per call** — most functions accept the parameter as a keyword argument,
2. **per process** — environment variables with the ``PLEQUE_`` prefix,
3. **persistently** — a ``pleque.toml`` configuration file (or a
   ``[tool.pleque]`` table in ``pyproject.toml``).

Configuration file
------------------

On first use PLEQUE looks for a configuration file in the following locations
and loads the **first one that exists** (configuration files are *not* merged —
the remaining locations are ignored):

.. list-table::
   :header-rows: 1
   :widths: 10 50 40

   * - Priority
     - Location
     - Notes
   * - 1
     - File pointed to by the ``PLEQUE_CONFIG_FILE`` environment variable
     - Raises ``FileNotFoundError`` if set but missing.
   * - 2
     - ``./pleque.toml``
     - Current working directory.
   * - 3
     - ``pleque.toml`` in the user configuration directory
     - Linux/macOS: ``$XDG_CONFIG_HOME/pleque/pleque.toml``
       (``~/.config/pleque/pleque.toml`` by default).
       Windows: ``%APPDATA%\pleque\pleque.toml``.
   * - 4
     - ``[tool.pleque]`` table in the nearest ``pyproject.toml``
     - Found by walking up from the current working directory; only counts
       if the ``[tool.pleque]`` table is actually present.
   * - 5
     - Built-in defaults
     - The values listed in the reference below.

Format example
--------------

``pleque.toml`` contains one TOML table per settings section; any subset of
keys may be given, everything else keeps its default:

.. code-block:: toml

   default_cocos = 3

   [flux_surfaces]
   n_psi = 300
   psi_n_min = 0.005

   [lcfs]
   search_grid_nr = 1000
   search_grid_nz = 1500

   [field_line_tracing]
   rtol = 1e-9

The equivalent form inside a project's ``pyproject.toml``:

.. code-block:: toml

   [tool.pleque]
   default_cocos = 3

   [tool.pleque.flux_surfaces]
   n_psi = 300

.. note::
   Unknown keys in a configuration file are silently ignored (this keeps old
   PLEQUE versions compatible with configs written for newer ones) — double
   check the spelling of the keys you set.

Environment variables
---------------------

Every setting can also be set through an environment variable with the
``PLEQUE_`` prefix; nested sections are addressed with a double underscore.
Environment variables take precedence over values from configuration files:

.. code-block:: bash

   export PLEQUE_DEFAULT_COCOS=3
   export PLEQUE_FLUX_SURFACES__N_PSI=300
   export PLEQUE_LCFS__REFINEMENT_TOLERANCE=1e-8

Programmatic access
-------------------

.. code-block:: python

   from pleque.config import get_settings, reload_settings

   settings = get_settings()           # cached, process-wide instance
   settings.flux_surfaces.n_psi        # -> 200

   settings.config_source              # Path of the loaded file, or None
   settings.config_files_consulted     # all locations checked, in order

   settings = reload_settings()        # re-read files/env (clears the cache)

``get_settings()`` is cached (``functools.lru_cache``), so PLEQUE reads the
configuration once per process; call ``reload_settings()`` after changing a
configuration file or environment variable at runtime. A fresh, uncached
instance can be built with ``pleque.config.load_settings()``, and ad-hoc
overrides can be passed directly: ``Settings(flux_surfaces={"n_psi": 50})``.

Settings reference
------------------

Top level
^^^^^^^^^

=================== ======= ========= ==========================================================
Name                Type    Default   Description
=================== ======= ========= ==========================================================
``default_cocos``   int     3         COCOS convention assumed when the input does not specify one.
``debug_plots``     bool    false     Draw (blocking) matplotlib debug plots when the equilibrium
                                      initialization fails. Keep disabled in headless environments.
=================== ======= ========= ==========================================================

``[grid]`` — default computational grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

==================================== ======= ========= ==========================================================
Name                                 Type    Default   Description
==================================== ======= ========= ==========================================================
``default_nr``                       int     1000      Number of R points of the default ``Equilibrium.grid()`` grid.
``default_nz``                       int     2000      Number of Z points of the default ``Equilibrium.grid()`` grid.
``synthetic_wall_margin``            float   0.01      Fractional inset of the rectangular first wall synthesized when no wall is given.
``synthetic_wall_points_per_side``   int     20        Number of points per side of the synthesized rectangular first wall.
==================================== ======= ========= ==========================================================

``[splines]`` — spline construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

==================== ======= ========= ==========================================================
Name                 Type    Default   Description
==================== ======= ========= ==========================================================
``psi_order``        int     3         Order of the 2D psi(R, Z) spline.
``psi_smooth``       float   0.0       Smoothing factor ``s`` of the 2D psi(R, Z) spline.
``profile_order``    int     3         Order ``k`` of 1D profile splines (F, pressure, q, ...).
``profile_smooth``   float   0.0       Smoothing factor ``s`` of 1D profile splines.
==================== ======= ========= ==========================================================

``[critical_points]`` — magnetic axis and X-point search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

====================================== ======= ========= ==========================================================
Name                                   Type    Default   Description
====================================== ======= ========= ==========================================================
``find_extremes_order``                int     20        Number of neighbouring grid points used by ``argrelmin`` when locating extremes of \|grad psi\|.
``find_extremes_max_iter``             int     10        Maximum number of retries (with reduced order) when no O-point candidate is found.
``gradient_threshold``                 float   1.0       Upper bound on \|grad psi\|^2 for a grid point to count as a critical-point candidate.
``vicinity_radius``                    float   0.1       Half-size [m] of the (R, Z) box used when refining a critical point by local minimization.
``minimizer_xtol``                     float   1e-7      ``xtol`` passed to the Powell/TNC minimizers refining extremes.
``relocation_threshold``               float   1e-2      If unbounded minimization moves a candidate point by more than this, the bounded minimizer is used instead.
``axis_vertical_weight``               float   5.0       Weight of the radial distance when ranking O-point candidates.
``axis_out_of_wall_penalty``           float   1e-3      Score multiplier penalizing O-point candidates outside the first wall.
``x_point_sort_epsilon``               float   1e-3      Regularization added to X-point ranking terms to avoid zero scores.
``monotonicity_test_points``           int     10        Number of test points for the psi-monotonicity check between magnetic axis and X-point.
``limiter_monotonicity_test_points``   int     50        Number of test points for the psi-monotonicity check between magnetic axis and limiter point.
====================================== ======= ========= ==========================================================

``[lcfs]`` — last closed flux surface search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================ ======= ========= ==========================================================
Name                         Type    Default   Description
============================ ======= ========= ==========================================================
``search_grid_nr``           int     700       Number of R points of the grid used to locate the LCFS contour.
``search_grid_nz``           int     1200      Number of Z points of the grid used to locate the LCFS contour.
``refinement_tolerance``     float   1e-10     Relative psi error under which the iterative LCFS refinement stops.
``inner_psi_n_offset``       float   1e-5      Offset used as psi_n = 1 - offset when contouring the LCFS as an inner flux surface.
``initial_psi_offset``       float   1e-4      Fraction of (psi_lcfs - psi_axis) used to shift psi inwards for the initial LCFS candidate.
``surface_step_damping``     float   0.99      Damping factor of the gradient step in the iterative flux-surface refinement.
``x_point_shift``            float   1e-6      Distance [m] by which the separatrix tracing start point is shifted from the X-point.
============================ ======= ========= ==========================================================

``[field_line_tracing]`` — field-line tracing (ODE solver)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================== ======= =========== ==========================================================
Name                           Type    Default     Description
============================== ======= =========== ==========================================================
``atol``                       float   1e-6        Absolute tolerance of the field-line ODE solver.
``rtol``                       float   1e-8        Relative tolerance of the field-line ODE solver.
``x_point_atol_scale``         float   1e-3        For X-point plasmas the absolute tolerance is reduced to ``min(atol, distance_to_x_point * x_point_atol_scale)``.
``max_step``                   float   1e-2        Maximum integration step in toroidal angle phi [rad].
``max_toroidal_turns``         int     50          Maximum number of toroidal turns integrated when tracing.
``poloidal_stop_resolution``   float   pi/1024     Poloidal-angle resolution [rad] of the stopper used by ``Equilibrium.trace_field_line``.
``default_stop_resolution``    float   pi/360      Default poloidal-angle resolution [rad] of ``poloidal_angle_stopper_factory``.
``target_stopper_atol``        float   1e-6        Distance tolerance [m] of the target-point stopper used when tracing flux surfaces.
============================== ======= =========== ==========================================================

``[flux_surfaces]`` — flux-surface generation and 1D profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========================= ======= ========= ==========================================================
Name                      Type    Default   Description
========================= ======= ========= ==========================================================
``n_psi``                 int     200       Default number of psi_n levels used to evaluate flux surfaces.
``psi_n_min``             float   0.01      Minimum psi_n used for flux-surface averaging and volume integration.
``contour_step``          float   1e-3      Spatial resolution [m] of flux-surface contours found on the (R, Z) grid.
``contour_grid_nr``       int     100       Default number of R points of the grid used for flux-surface contouring.
``contour_grid_nz``       int     100       Default number of Z points of the grid used for flux-surface contouring.
``trace_step``            float   1e-3      Maximum step [m] along the flux surface used by ``Equilibrium.trace_flux_surface``.
``trace_max_turns``       int     4         Maximum number of toroidal turns integrated when tracing a flux surface.
``q_psi_n_min``           float   0.01      Lowest psi_n of the grid on which the q profile is evaluated.
``q_psi_n_step``          float   0.005     psi_n step of the grid on which the q profile is evaluated.
``q_search_psi_n_max``    float   0.95      Default upper psi_n bound when searching a flux surface with a given q.
``q_search_psi_n_cap``    float   0.99      Hard upper psi_n bound of the root finder used when searching a flux surface with a given q.
``midplane_map_points``   int     100       Number of points of the midplane r_mid -> psi mapping spline.
========================= ======= ========= ==========================================================

``[plotting]`` — plotting helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================ ======= ========= ==========================================================
Name                         Type    Default   Description
============================ ======= ========= ==========================================================
``grid_nr``                  int     400       Number of R points of grids used by plotting routines.
``grid_nz``                  int     600       Number of Z points of grids used by plotting routines.
``debug_contour_levels``     int     60        Number of psi contour levels in the debug overview plot.
``psi_contour_levels``       int     20        Default number of contour levels of ``plot_psi_contours``.
``rational_q_psi_n_max``     float   0.95      Upper psi_n bound when looking up rational q surfaces for plotting.
``sol_spacing``              float   2e-3      Default radial spacing [m] of near-SOL contours.
``axis_margin``              float   1/12      Margin around the first wall, as a fraction of its size, used for axis limits.
============================ ======= ========= ==========================================================

``[io]`` — file writers
^^^^^^^^^^^^^^^^^^^^^^^

==================== ======= ========= ==========================================================
Name                 Type    Default   Description
==================== ======= ========= ==========================================================
``geqdsk_nx``        int     64        Default number of R points of a written G-EQDSK file.
``geqdsk_ny``        int     128       Default number of Z points of a written G-EQDSK file.
``geqdsk_nbdry``     int     200       Default number of boundary points of a written G-EQDSK file.
``omas_n_psi``       int     200       Default number of 1D profile points written to an OMAS data structure.
``omas_grid_step``   float   1e-3      Default (R, Z) grid step [m] of 2D profiles written to an OMAS data structure.
==================== ======= ========= ==========================================================

Migration from older versions
-----------------------------

Up to PLEQUE 0.0.8 the (undocumented) ``Settings`` class had four flat fields.
They moved into sections:

==================== ================================
Old name             New name
==================== ================================
``npsi_grid``        ``flux_surfaces.n_psi``
``psin0``            ``flux_surfaces.psi_n_min``
``nr_grid``          ``lcfs.search_grid_nr``
``nz_grid``          ``lcfs.search_grid_nz``
==================== ================================

``nr_grid``/``nz_grid`` previously defaulted to ``None`` ("use the fallback
700/1200"); the new fields default to 700/1200 directly. Unprefixed
environment variables (e.g. ``NPSI_GRID``) are no longer read — use the
``PLEQUE_`` prefix.


.. _logging:

Logging
-------

PLEQUE logs through the standard :mod:`logging` machinery under the
``pleque`` logger hierarchy (one logger per module, e.g.
``pleque.core.equilibrium``). As a library, PLEQUE does not configure any
handler by default; to see the messages either use the convenience helper

.. code-block:: python

   import logging
   import pleque

   pleque.set_log_level(logging.DEBUG)  # or logging.INFO

or configure the logger yourself:

.. code-block:: python

   import logging

   logging.basicConfig()
   logging.getLogger("pleque").setLevel(logging.INFO)

The ``verbose=True`` argument of :class:`pleque.Equilibrium` is kept for
backward compatibility and simply calls
``pleque.set_log_level(logging.DEBUG)``.
