"""Soil moisture loss models: analytical solutions and their ODEs.

Symbols and units (shared by every function):

    t          time since the event start                           [d]
    theta      volumetric soil moisture                             [m3 m-3]
    theta_0    theta at t = 0                                       [m3 m-3]
    theta_w    wilting point (loss stops)                           [m3 m-3]
    theta_star critical soil moisture (loss is k at and above it)   [m3 m-3]
    k          maximum loss rate, k = ET_max / (1000 dz)            [m3 m-3 d-1]
    q          shape exponent of the loss function                  [-]
    tau        e-folding time of the exponential model              [d]

Identifiability. In ``q_stage2`` the parameters k and theta_star enter the
solution only through the combination k / (theta_star - theta_w)^q. With
theta_0 and theta_w fixed (Morgan's setting), k and theta_star are therefore
not separately identifiable from a single drydown: the fit determines q and
that ratio, and the split between k and theta_star is set by the starting
values and bounds. Report k and theta_star with this in mind.

Each ``*_rate`` function is the right-hand side of the ODE; the solutions are
checked against numerical integration of these in tests/test_models.py.
"""

import numpy as np


def exponential(t_days, theta_0, theta_w, tau_days):
    """Exponential drydown (McColl et al. 2017).

        theta(t) = (theta_0 - theta_w) * exp(-t / tau) + theta_w

    Solution of dtheta/dt = -(theta - theta_w) / tau. Units in module docstring.
    """
    return (theta_0 - theta_w) * np.exp(-np.asarray(t_days, dtype=float) / tau_days) + theta_w


def exponential_rate(theta, theta_w, tau_days):
    """dtheta/dt = -(theta - theta_w) / tau   [m3 m-3 d-1]."""
    return -(theta - theta_w) / tau_days


def q_stage2(t_days, theta_0, theta_w, k_per_day, q, theta_star):
    """Nonlinear stage-2 drydown (the "q model" of Morgan et al. 2025).

        dtheta/dt = -k * ((theta - theta_w) / (theta_star - theta_w))^q

        theta(t) = [ (theta_0 - theta_w)^(1-q)
                     - k (1-q) / (theta_star - theta_w)^q * t ]^(1/(1-q)) + theta_w

    For q = 1 exactly the limit is used:
        theta(t) = theta_w + (theta_0 - theta_w) * exp(-k t / (theta_star - theta_w)).

    For q < 1 the bracket reaches zero (theta = theta_w) in finite time and is
    negative afterwards. The power then gives NaN, or, when 1/(1-q) is an
    integer (e.g. q = 0.5), a real value that rises again past theta_w. Both
    are Morgan's ``q_model`` behaviour and are kept. This form applies the same law above theta_star (loss > k
    there); ``q_piecewise`` caps the loss at k. Units in module docstring.
    """
    t = np.asarray(t_days, dtype=float)
    s = theta_star - theta_w
    if q == 1:
        return theta_w + (theta_0 - theta_w) * np.exp(-k_per_day * t / s)
    with np.errstate(invalid="ignore", divide="ignore"):
        bracket = (theta_0 - theta_w) ** (1 - q) - k_per_day * (1 - q) / s ** q * t
        return bracket ** (1 / (1 - q)) + theta_w


def q_stage2_rate(theta, theta_w, k_per_day, q, theta_star):
    """dtheta/dt = -k * ((theta - theta_w) / (theta_star - theta_w))^q   [m3 m-3 d-1]."""
    x = np.maximum((theta - theta_w) / (theta_star - theta_w), 0)
    return -k_per_day * x ** q


def q_piecewise(t_days, theta_0, theta_w, k_per_day, q, theta_star):
    """Stage 1 (constant rate k) above theta_star, then stage 2 (``q_stage2``).

        dtheta/dt = -k                                          theta >= theta_star
        dtheta/dt = -k * ((theta - theta_w)/(theta_star - theta_w))^q   theta < theta_star

        t_star = max(theta_0 - theta_star, 0) / k               [d]
        theta(t) = theta_0 - k t                                t < t_star
        theta(t) = q_stage2(t - t_star; theta_0 = min(theta_0, theta_star))   t >= t_star

    Unlike Morgan's ``q_model_piecewise`` (which restarts stage 2 from theta_0),
    stage 2 here starts from theta_star at t_star, so theta(t) is continuous.
    """
    t = np.asarray(t_days, dtype=float)
    t_star = max(theta_0 - theta_star, 0) / k_per_day
    stage2 = q_stage2(np.maximum(t - t_star, 0), min(theta_0, theta_star),
                      theta_w, k_per_day, q, theta_star)
    return np.where(t < t_star, theta_0 - k_per_day * t, stage2)


def q_piecewise_rate(theta, theta_w, k_per_day, q, theta_star):
    """-k above theta_star, ``q_stage2_rate`` below   [m3 m-3 d-1]."""
    return np.where(theta >= theta_star, -k_per_day,
                    q_stage2_rate(theta, theta_w, k_per_day, q, theta_star))


MODELS = {"exponential": exponential, "q_stage2": q_stage2, "q_piecewise": q_piecewise}
