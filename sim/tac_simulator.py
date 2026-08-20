# sim/tac_simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import nnls

from kinetics.helpers.twotcm import simulate_2tcm
from core.noise import add_noise_voxel
from utils.frame_average import frame_average
from separation.base import SeparationAlgorithm, SeparationResult
from kinetics.helpers.basis_gamma import gamma_shape


@dataclass
class DualTacSimulation:
    t_frames: np.ndarray
    t_int: np.ndarray
    delta_min: float

    Cp1: np.ndarray
    Cp2: np.ndarray

    Ct1_bio_tavg_n0_d0: np.ndarray
    Ct2_bio_tavg_n0_d0: np.ndarray

    Ct1_phys_tavg_n0_d1: np.ndarray
    Ct2_phys_tavg_n0_d1: np.ndarray

    Ct1_bio_tint_n0_d0: np.ndarray # we need the curves on internal timelines as well for smooth plotting purposes
    Ct2_bio_tint_n0_d0: np.ndarray
    Ct1_phys_tint_n0_d1: np.ndarray
    Ct2_phys_tint_n0_d1: np.ndarray

    y_clean_tavg_n0_d1: np.ndarray
    y_clean_tint_n0_d1: np.ndarray
    y_meas_tavg_n1_d1: np.ndarray
    expected_counts: np.ndarray

    decay1_tavg: np.ndarray
    decay2_tavg: np.ndarray
    t_rel1: np.ndarray
    t_rel2: np.ndarray
    lam1: float
    lam2: float


@dataclass
class DualTacResult:
    t_frames: np.ndarray
    t_int: np.ndarray
    Delta: float

    Cp1: np.ndarray
    Cp2: np.ndarray

    Ct1_bio_tint_n0_d0: np.ndarray  #true biological curve for tracer 1, internal time grid
    Ct2_bio_tint_n0_d0: np.ndarray  #true biological curve for tracer 2, internal time grid
    Ct1_phys_tint_n0_d1: np.ndarray #true biological curve for tracer 1, internal time grid, decayed
    Ct2_phys_tint_n0_d1: np.ndarray #true biological curve for tracer 2, internal time grid, decayed

    Ct1_bio_tavg_n0_d0: np.ndarray #true biological curve for tracer 1, frame-averaged
    Ct2_bio_tavg_n0_d0: np.ndarray #true biological curve for tracer 2, frame-averaged

    Ct1_phys_tavg_n0_d1: np.ndarray #true physical curve for tracer 1, frame-averaged, decay-corrected
    Ct2_phys_tavg_n0_d1: np.ndarray #true physical curve for tracer 2, frame-averaged, decay-corrected

    y_clean_tavg_n0_d1: np.ndarray  #clean frame-level measured-domain sum: Ct1_phys_tavg_n0_d1 + Ct2_phys_tavg_n0_d1
    y_meas_tavg_n1_d1: np.ndarray  #noisy frame-level measured data

   

    Ct1_est_bio_tavg_n1_d0: np.ndarray # estimated biological TAC for tracer 1 on frame grid (because we assume it's realistic measurements)
    Ct2_est_bio_tavg_n1_d0: np.ndarray # estimated biological TAC for tracer 2 on frame grid (because we assume it's realistic measurements)

    ct_1_est_meas_tavg_n1_d1: np.ndarray # estimated measured-domain tracer 1 TAC on frame grid (because we assume it's realistic measurements)
    ct_2_est_meas_tavg_n1_d1: np.ndarray # estimated measured-domain tracer 2 TAC on frame grid (because we assume it's realistic measurements)

    ct_1_est_meas_tint_n1_d1: np.ndarray # reconstruction of the curves using the internal time grid as opposed to frame grids, to make a smooth curve
    ct_2_est_meas_tint_n1_d1: np.ndarray
    Ct1_est_bio_tint_n1_d0: np.ndarray  # decay-corrected reconstruction of the curves using the internal time grid as opposed to frame grids, to make a smooth curve
    Ct2_est_bio_tint_n1_d0: np.ndarray

    nrmse_est_1_bio: float
    nrmse_est_2_bio: float

    coef_1: np.ndarray | None = None 
    coef_2: np.ndarray | None = None


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) + eps)
    )


def make_tracer_bases(
    Ct1_bio_tavg_n0_d0: np.ndarray,
    Ct2_bio_tavg_n0_d0: np.ndarray,
    gamma_lib_1: np.ndarray,
    Gamma_1: np.ndarray,
    gamma_lib_2: np.ndarray,
    Gamma_2: np.ndarray,
    decay1: np.ndarray,
    decay2: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit each biological TAC to its gamma library, keep only the active basis
    functions, then apply physical decay to create measured-domain bases.
    """
    w1, _ = nnls(Gamma_1, Ct1_bio_tavg_n0_d0)
    w2, _ = nnls(Gamma_2, Ct2_bio_tavg_n0_d0)

    idx1 = np.where(w1 > eps)[0]
    idx2 = np.where(w2 > eps)[0]

    psi1 = gamma_lib_1[idx1]
    psi2 = gamma_lib_2[idx2]

    phi1 = psi1 * decay1[np.newaxis, :]
    phi2 = psi2 * decay2[np.newaxis, :]

    return phi1.T, phi2.T, idx1, idx2


def simulate_dual_tracer_measurement(
        timegrid: Any, 
        tracer1: Any,
        tracer2: Any, 
        protocol: Any, 
        rng=None,
        ) -> DualTacSimulation:
    """
    Simulate biological and physical TACs for both tracers and add measurement noise.
    """
    t_int = timegrid.t_internal
    t_frames = timegrid.frame_mids
    delta_min = protocol.delta_min

    Cp1_int = tracer1.aif(t_int)
    Cp2_int = tracer2.aif(t_int, delta_min=delta_min)

    Ct1_int, _, _ = simulate_2tcm(**tracer1.true_params(), Cp=Cp1_int, t=t_int)
    Ct2_int, _, _ = simulate_2tcm(**tracer2.true_params(), Cp=Cp2_int, t=t_int)

    Ct1_bio_tint_n0_d0 = Ct1_int #need curves on internal timeframes for smooth plotting purposes
    Ct2_bio_tint_n0_d0 = Ct2_int

    Ct1_bio_tavg_n0_d0 = frame_average(t_int, Ct1_int, timegrid.frame_edges)
    Ct2_bio_tavg_n0_d0 = frame_average(t_int, Ct2_int, timegrid.frame_edges)

    lam1 = np.log(2.0) / tracer1.half_life_min
    lam2 = np.log(2.0) / tracer2.half_life_min

    t_rel1_int = t_int  #once again anything with _int here is because we need curves in internal frames for smooth plotting purposes
    t_rel2_int = np.maximum(t_int - delta_min, 0.0)

    decay1_int = np.exp(-lam1 * t_rel1_int)
    decay2_int = np.exp(-lam2 * t_rel2_int)

    Ct1_phys_tint_n0_d1 = Ct1_bio_tint_n0_d0 * decay1_int
    Ct2_phys_tint_n0_d1 = Ct2_bio_tint_n0_d0 * decay2_int

    t_rel1 = t_frames
    t_rel2 = np.maximum(t_frames - delta_min, 0.0)

    decay1_tavg = frame_average(t_int, decay1_int, timegrid.frame_edges)
    decay2_tavg = frame_average(t_int, decay2_int, timegrid.frame_edges)

    Ct1_phys_tavg_n0_d1 = frame_average(t_int, Ct1_phys_tint_n0_d1, timegrid.frame_edges)
    Ct2_phys_tavg_n0_d1 = frame_average(t_int, Ct2_phys_tint_n0_d1, timegrid.frame_edges)
    y_clean_tint_n0_d1 = Ct1_phys_tint_n0_d1 + Ct2_phys_tint_n0_d1
    y_clean_tavg_n0_d1 = Ct1_phys_tavg_n0_d1 + Ct2_phys_tavg_n0_d1

    y_meas_tavg_n1_d1, expected_counts = add_noise_voxel(
        y_clean_tavg_n0_d1,
        timegrid.frame_durs,
        voxel_size_mm=4.0,
        scanner_name="panorama_gs",
        rng=rng,
    )

    return DualTacSimulation(
        t_frames=t_frames,
        t_int = t_int,
        delta_min=delta_min,
        Cp1=Cp1_int,
        Cp2=Cp2_int,
        Ct1_bio_tavg_n0_d0=Ct1_bio_tavg_n0_d0,
        Ct2_bio_tavg_n0_d0=Ct2_bio_tavg_n0_d0,
        Ct1_phys_tavg_n0_d1=Ct1_phys_tavg_n0_d1,
        Ct2_phys_tavg_n0_d1=Ct2_phys_tavg_n0_d1,
        Ct1_bio_tint_n0_d0=Ct1_bio_tint_n0_d0, #anything with _int is for smooth plotting purposes
        Ct2_bio_tint_n0_d0=Ct2_bio_tint_n0_d0,
        Ct1_phys_tint_n0_d1=Ct1_phys_tint_n0_d1,
        Ct2_phys_tint_n0_d1=Ct2_phys_tint_n0_d1,
        y_clean_tavg_n0_d1=y_clean_tavg_n0_d1,
        y_clean_tint_n0_d1=y_clean_tint_n0_d1,
        y_meas_tavg_n1_d1=y_meas_tavg_n1_d1,
        expected_counts=expected_counts,
        decay1_tavg=decay1_tavg,
        decay2_tavg=decay2_tavg,
        t_rel1=t_rel1,
        t_rel2=t_rel2,
        lam1=lam1,
        lam2=lam2,
    )


def estimate_biological_curves(
    tracer1_meas: np.ndarray,
    tracer2_meas: np.ndarray,
    decay1_tavg: np.ndarray,
    decay2_tavg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decay-correct frame-averaged measured curves with frame-integrated factors.
    """
    Ct1_est_bio_tavg_n1_d0 = tracer1_meas / decay1_tavg
    Ct2_est_bio_tavg_n1_d0 = tracer2_meas / decay2_tavg
    return Ct1_est_bio_tavg_n1_d0, Ct2_est_bio_tavg_n1_d0


def simulate_dual_tac_any_alg(
    timegrid,
    rac,
    fdg,
    protocol,
    gamma_lib_1: np.ndarray,
    Gamma_1: np.ndarray,
    gamma_params_1: list[tuple[float, float]],
    scale_1: float,
    gamma_lib_2: np.ndarray,
    Gamma_2: np.ndarray,
    gamma_params_2: list[tuple[float,float]],
    scale_2: float,
    rng,
    separation_alg: SeparationAlgorithm,
) -> DualTacResult:
    """
    Simulate a noisy dual-tracer TAC and separate it with an arbitrary algorithm.
    """
    sim = simulate_dual_tracer_measurement(
        timegrid=timegrid,
        tracer1=rac,
        tracer2=fdg,
        protocol=protocol,
        rng=rng,
    )

    Phi_rac, Phi_fdg, idx1, idx2 = make_tracer_bases(
        sim.Ct1_bio_tavg_n0_d0,
        sim.Ct2_bio_tavg_n0_d0,
        gamma_lib_1,
        Gamma_1,
        gamma_lib_2,
        Gamma_2,
        sim.decay1_tavg,
        sim.decay2_tavg,
    )

    separation_result: SeparationResult = separation_alg.separate(
        y=sim.y_meas_tavg_n1_d1,
        t_frames = sim.t_frames,
        Phi_1= Phi_rac,
        Phi_2= Phi_fdg,
    )


    Ct1_est_bio_tavg_n1_d0, Ct2_est_bio_tavg_n1_d0 = estimate_biological_curves(
        tracer1_meas=separation_result.tracer1_curve,
        tracer2_meas=separation_result.tracer2_curve,
        decay1_tavg=sim.decay1_tavg,
        decay2_tavg=sim.decay2_tavg,
    )

    active_params_1 = [gamma_params_1[i] for i in idx1]
    active_params_2 = [gamma_params_2[i] for i in idx2]

    psi1_int = np.array([
        gamma_shape(sim.t_int, t0, tau, scale = scale_1)
        for t0, tau in active_params_1])
    
    psi2_int = np.array([
        gamma_shape(sim.t_int, t0, tau, scale=scale_2)
        for t0, tau in active_params_2
    ])

    t_rel1_int = sim.t_int
    t_rel2_int = np.maximum(sim.t_int - sim.delta_min, 0.0)

    decay1_int = np.exp(-sim.lam1 * t_rel1_int)
    decay2_int = np.exp(-sim.lam2 * t_rel2_int)

    Phi1_int = (psi1_int * decay1_int[np.newaxis, :]).T
    Phi2_int = (psi2_int * decay2_int[np.newaxis, :]).T

    ct_1_est_meas_tint_n1_d1 = Phi1_int @ separation_result.coef_1
    ct_2_est_meas_tint_n1_d1 = Phi2_int @ separation_result.coef_2

    Ct1_est_bio_tint_n1_d0 = ct_1_est_meas_tint_n1_d1 * np.exp(sim.lam1 * t_rel1_int)
    Ct2_est_bio_tint_n1_d0 = ct_2_est_meas_tint_n1_d1 * np.exp(sim.lam2 * t_rel2_int)

    return DualTacResult(
        t_frames=sim.t_frames,
        t_int = sim.t_int,
        Delta= sim.delta_min,
        Cp1=sim.Cp1,
        Cp2=sim.Cp2,
        Ct1_bio_tavg_n0_d0=sim.Ct1_bio_tavg_n0_d0,
        Ct2_bio_tavg_n0_d0=sim.Ct2_bio_tavg_n0_d0,
        Ct1_bio_tint_n0_d0 = sim.Ct1_bio_tint_n0_d0,
        Ct2_bio_tint_n0_d0 = sim.Ct2_bio_tint_n0_d0,
        Ct1_phys_tavg_n0_d1=sim.Ct1_phys_tavg_n0_d1,
        Ct2_phys_tavg_n0_d1=sim.Ct2_phys_tavg_n0_d1,
        Ct1_phys_tint_n0_d1 = sim.Ct1_phys_tint_n0_d1,
        Ct2_phys_tint_n0_d1 = sim.Ct2_phys_tint_n0_d1,
        y_clean_tavg_n0_d1=sim.y_clean_tavg_n0_d1,
        y_meas_tavg_n1_d1=sim.y_meas_tavg_n1_d1,
        Ct1_est_bio_tavg_n1_d0=Ct1_est_bio_tavg_n1_d0,
        Ct2_est_bio_tavg_n1_d0=Ct2_est_bio_tavg_n1_d0,
        ct_1_est_meas_tavg_n1_d1=separation_result.tracer1_curve,
        ct_2_est_meas_tavg_n1_d1=separation_result.tracer2_curve,
        ct_1_est_meas_tint_n1_d1=ct_1_est_meas_tint_n1_d1,
        ct_2_est_meas_tint_n1_d1=ct_2_est_meas_tint_n1_d1,
        Ct1_est_bio_tint_n1_d0=Ct1_est_bio_tint_n1_d0,
        Ct2_est_bio_tint_n1_d0=Ct2_est_bio_tint_n1_d0,
        coef_1 = separation_result.coef_1,
        coef_2 = separation_result.coef_2,
        nrmse_est_1_bio=nrmse(sim.Ct1_bio_tavg_n0_d0, Ct1_est_bio_tavg_n1_d0),
        nrmse_est_2_bio=nrmse(sim.Ct2_bio_tavg_n0_d0, Ct2_est_bio_tavg_n1_d0),
    )


def simulate_dual_tac_legacy(
    timegrid,
    rac,
    fdg,
    protocol,
    gamma_lib_1: np.ndarray,
    Gamma_1: np.ndarray,
    gamma_lib_2: np.ndarray,
    Gamma_2: np.ndarray,
    rng,
    use_joint: bool = False,
) -> dict[str, Any]:
    """
    Thin compatibility wrapper for the older non-registry path.
    """
    sim = simulate_dual_tracer_measurement(
        timegrid=timegrid,
        rac=rac,
        fdg=fdg,
        protocol=protocol,
        rng=rng,
    )

    Phi_rac, Phi_fdg = make_tracer_bases(
        sim.Ct1_bio_tavg_n0_d0,
        sim.Ct2_bio_tavg_n0_d0,
        gamma_lib_1,
        Gamma_1,
        gamma_lib_2,
        Gamma_2,
        sim.decay1_tavg,
        sim.decay2_tavg,
    )

    if use_joint:
        from separation.sequential_nnls import joint_unmix

        rac_est_phys, fdg_est_phys, _ = joint_unmix(
            sim.y_clean_tavg_n0_d1,
            Phi_rac,
            Phi_fdg,
            sim.t_frames,
            t_cut=protocol.early_cut(),
        )
    else:
        from separation.sequential_nnls import sequential_unmix_l2

        rac_est_phys, fdg_est_phys = sequential_unmix_l2(
            sim.y_meas_tavg_n1_d1,
            Phi_rac,
            Phi_fdg,
            sim.t_frames,
            protocol.early_cut(),
            alpha=0.3,
        )

    Ct1_est_bio_tavg_n1_d0, Ct2_est_bio_tavg_n1_d0 = estimate_biological_curves(
        tracer1_meas=rac_est_phys,
        tracer2_meas=fdg_est_phys,
        decay1_tavg=sim.decay1_tavg,
        decay2_tavg=sim.decay2_tavg,
    )

    return {
        "timegrid" : timegrid,
        "t_frames": sim.t_frames,
        "Delta": sim.Delta,
        "Cp1": sim.Cp1,
        "Cp2": sim.Cp2,
        "Ct1_bio_tavg_n0_d0": sim.Ct1_bio_tavg_n0_d0,
        "Ct2_bio_tavg_n0_d0": sim.Ct2_bio_tavg_n0_d0,
        "Ct1_phys_tavg_n0_d1": sim.Ct1_phys_tavg_n0_d1,
        "Ct2_phys_tavg_n0_d1": sim.Ct2_phys_tavg_n0_d1,
        "y_clean_tavg_n0_d1": sim.y_clean_tavg_n0_d1,
        "y_meas_tavg_n1_d1": sim.y_meas_tavg_n1_d1,
        "Ct1_est_bio_tavg_n1_d0": Ct1_est_bio_tavg_n1_d0,
        "Ct2_est_bio_tavg_n1_d0": Ct2_est_bio_tavg_n1_d0,
    }
# sim/tac_simulator.py
# import numpy as np
# from kinetics.helpers.twotcm import simulate_2tcm
# from core.noise import add_noise_voxel, add_noise_roi
# from utils.frame_average import frame_average
# from scipy.optimize import nnls
# from separation.base import SeparationAlgorithm, SeparationResult

# def make_tracer_bases(Ct1_bio_tavg_n0_d0, Ct2_bio_tavg_n0_d0, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2, eps=1e-6):
#     w1, _ = nnls(Gamma_1, Ct1_bio_tavg_n0_d0)
#     w2, _ = nnls(Gamma_2, Ct2_bio_tavg_n0_d0) #first fit the biological curves to the gamma basis

#     idx1 = np.where(w1 > eps)[0] #only keep the basis functions that contribute significantly, this step is only done in simulation to speed things up, i
#     #in real data we would not know this and would have to use the full basis
#     idx2 = np.where(w2 > eps)[0]

#     psi1 = gamma_lib_1[idx1] 
#     psi2 = gamma_lib_2[idx2]

#     phi1 = psi1 * decay1[np.newaxis, :] #gnerate the basis functions scaled by the decay
#     phi2 = psi2 * decay2[np.newaxis, :]

#     return phi1.T, phi2.T
# def simulate_dual_tac_any_alg(timegrid, rac, fdg, protocol, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, rng, separation_alg: SeparationAlgorithm, context = None):
#     t_int = timegrid.t_internal
#     t_frames = timegrid.frame_mids
#     Delta = protocol.delta_min

#     Cp1_int = rac.aif(t_int)
#     Cp2_int = fdg.aif(t_int, delta_min=Delta)

#     Ct1_int, _, _ = simulate_2tcm(**rac.true_params(), Cp=Cp1_int, t=t_int)
#     Ct2_int, _, _ = simulate_2tcm(**fdg.true_params(), Cp=Cp2_int, t=t_int)

#     Ct1_bio_tavg_n0_d0 = frame_average(t_int, Ct1_int, timegrid.frame_edges)
#     Ct2_bio_tavg_n0_d0 = frame_average(t_int, Ct2_int, timegrid.frame_edges)

#     lam1 = np.log(2) / rac.half_life_min
#     lam2 = np.log(2) / fdg.half_life_min
#     t_rel1 = t_frames
#     t_rel2 = np.maximum(t_frames - Delta, 0)

#     decay1 = np.exp(-lam1 * t_rel1)
#     decay2 = np.exp(-lam2 * t_rel2)

#     Ct1_phys_tavg_n0_d1 = Ct1_bio_tavg_n0_d0 * decay1
#     Ct2_phys_tavg_n0_d1 = Ct2_bio_tavg_n0_d0 * decay2
#     y_clean_tavg_n0_d1 = Ct1_phys_tavg_n0_d1 + Ct2_phys_tavg_n0_d1

#     y_meas_tavg_n1_d1 =  add_noise_voxel(y_clean_tavg_n0_d1, timegrid.frame_durs, voxel_size_mm=4.0,
#                     scanner_name="panorama_gs", rng=None)[0]

#     Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio_tavg_n0_d0, Ct2_bio_tavg_n0_d0, gamma_lib_1, Gamma_1,gamma_lib_2, Gamma_2, decay1, decay2)
#     context["Phi_1"] = Phi_rac
#     context["Phi_2"] = Phi_fdg
#     context["t_frames"] = context.get("t_frames", t_frames)
#     context["t_cut"] = context.get("t_cut", 10.0)
#     context["alpha"] = context.get("alpha", 1.0)
#     separation_result: SeparationResult = separation_alg.separate(
#         y=y_meas_tavg_n1_d1,
#         x=None,
#         t=t_frames,
#         context=context
#     )

#     Ct1_est_bio_tavg_n1_d0 = separation_result.tracer1_curve * np.exp(+lam1 * t_rel1)
#     Ct2_est_bio_tavg_n1_d0 = separation_result.tracer2_curve * np.exp(+lam2 * t_rel2)

#     nrmse_est_1_bio = np.sqrt(np.mean((Ct1_bio_tavg_n0_d0 - Ct1_est_bio_tavg_n1_d0)**2)) / (np.max(Ct1_bio_tavg_n0_d0) + 1e-6)
#     nrmse_est_2_bio = np.sqrt(np.mean((Ct2_bio_tavg_n0_d0 - Ct2_est_bio_tavg_n1_d0)**2)) / (np.max(Ct2_bio_tavg_n0_d0) + 1e-6)

#     return dict(
#         t_frames=t_frames,
#         Delta=Delta,
#         Cp1=Cp1_int,
#         Cp2=Cp2_int,
#         Ct1_bio_tavg_n0_d0=Ct1_bio_tavg_n0_d0,
#         Ct2_bio_tavg_n0_d0=Ct2_bio_tavg_n0_d0,
#         Ct1_phys_tavg_n0_d1=Ct1_phys_tavg_n0_d1,
#         Ct2_phys_tavg_n0_d1=Ct2_phys_tavg_n0_d1,
#         y_clean_tavg_n0_d1=y_clean_tavg_n0_d1,
#         y_meas_tavg_n1_d1=y_meas_tavg_n1_d1,
#         Ct1_est_bio_tavg_n1_d0=Ct1_est_bio_tavg_n1_d0,
#         Ct2_est_bio_tavg_n1_d0=Ct2_est_bio_tavg_n1_d0,
#         ct_1_est_meas_tavg_n1_d1 = separation_result.tracer1_curve,
#         ct_2_est_meas_tavg_n1_d1 = separation_result.tracer2_curve,
#         nrmse_est_1_bio=nrmse_est_1_bio,
#         nrmse_est_2_bio=nrmse_est_2_bio
#     )


# def simulate_dual_tac(timegrid, rac, fdg, protocol, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, rng, use_joint=False):
#     t_int = timegrid.t_internal
#     t_frames = timegrid.frame_mids
#     Delta = protocol.delta_min

#     Cp1_int = rac.aif(t_int)
#     Cp2_int = fdg.aif(t_int, delta_min=Delta)

#     Ct1_int, _, _ = simulate_2tcm(**rac.true_params(), Cp=Cp1_int, t=t_int)
#     Ct2_int, _, _ = simulate_2tcm(**fdg.true_params(), Cp=Cp2_int, t=t_int)

#     Ct1_bio_tavg_n0_d0 = frame_average(t_int, Ct1_int, timegrid.frame_edges)
#     Ct2_bio_tavg_n0_d0 = frame_average(t_int, Ct2_int, timegrid.frame_edges)

#     lam1 = np.log(2) / rac.half_life_min
#     lam2 = np.log(2) / fdg.half_life_min
#     t_rel1 = t_frames
#     t_rel2 = np.maximum(t_frames - Delta, 0)

#     decay1 = np.exp(-lam1 * t_rel1)
#     decay2 = np.exp(-lam2 * t_rel2)

#     Ct1_phys_tavg_n0_d1 = Ct1_bio_tavg_n0_d0 * decay1
#     Ct2_phys_tavg_n0_d1 = Ct2_bio_tavg_n0_d0 * decay2
#     y_clean_tavg_n0_d1 = Ct1_phys_tavg_n0_d1 + Ct2_phys_tavg_n0_d1

#     y_meas_tavg_n1_d1 =  add_noise_voxel(y_clean_tavg_n0_d1, timegrid.frame_durs, voxel_size_mm=4.0,
#                     scanner_name="panorama_gs", rng=None)[0]

#     Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio_tavg_n0_d0, Ct2_bio_tavg_n0_d0, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2)
#     # Check the condition number of the basis matrices
#     # cond_rac = np.linalg.cond(Phi_rac)
#     # cond_fdg = np.linalg.cond(Phi_fdg)
#     # print(f"Condition number of PBR basis: {cond_rac:.2e}")
#     # print(f"Condition number of FDG basis: {cond_fdg:.2e}") 

#     if use_joint:
#         from separation.sequential_nnls import joint_unmix
#         rac_est_phys, fdg_est_phys, info = joint_unmix(y_clean_tavg_n0_d1, Phi_rac, Phi_fdg, t_frames, t_cut=protocol.early_cut())
#     else:
#         from separation.sequential_nnls import sequential_unmix_l2
#         t_cut = protocol.early_cut()
#         rac_est_phys, fdg_est_phys = sequential_unmix_l2(y_meas_tavg_n1_d1, Phi_rac, Phi_fdg, t_frames, t_cut, alpha=0.3)

#     Ct1_est_bio_tavg_n1_d0 = rac_est_phys * np.exp(+lam1 * t_rel1)
#     Ct2_est_bio_tavg_n1_d0 = fdg_est_phys * np.exp(+lam2 * t_rel2)

#     return dict(
#         t_frames=t_frames,
#         Delta=Delta,
#         Cp1=Cp1_int,
#         Cp2=Cp2_int,
#         Ct1_bio_tavg_n0_d0=Ct1_bio_tavg_n0_d0,
#         Ct2_bio_tavg_n0_d0=Ct2_bio_tavg_n0_d0,
#         Ct1_phys_tavg_n0_d1=Ct1_phys_tavg_n0_d1,
#         Ct2_phys_tavg_n0_d1=Ct2_phys_tavg_n0_d1,
#         y_clean_tavg_n0_d1=y_clean_tavg_n0_d1,
#         y_meas_tavg_n1_d1=y_meas_tavg_n1_d1,
#         Ct1_est_bio_tavg_n1_d0=Ct1_est_bio_tavg_n1_d0,
#         Ct2_est_bio_tavg_n1_d0=Ct2_est_bio_tavg_n1_d0
#     )
