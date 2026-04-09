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


@dataclass
class DualTacSimulation:
    t_frames: np.ndarray
    delta_min: float

    Cp1: np.ndarray
    Cp2: np.ndarray

    Ct1_bio: np.ndarray
    Ct2_bio: np.ndarray

    Ct1_phys: np.ndarray
    Ct2_phys: np.ndarray

    y_clean: np.ndarray
    y_meas: np.ndarray

    decay1: np.ndarray
    decay2: np.ndarray
    t_rel1: np.ndarray
    t_rel2: np.ndarray
    lam1: float
    lam2: float


@dataclass
class DualTacResult:
    t_frames: np.ndarray
    Delta: float

    Cp1: np.ndarray
    Cp2: np.ndarray

    Ct1_bio: np.ndarray
    Ct2_bio: np.ndarray

    Ct1_phys: np.ndarray
    Ct2_phys: np.ndarray

    y_clean: np.ndarray
    y_meas: np.ndarray

    Ct1_est_bio: np.ndarray
    Ct2_est_bio: np.ndarray

    ct_1_est_meas: np.ndarray
    ct_2_est_meas: np.ndarray

    nrmse_est_1_bio: float
    nrmse_est_2_bio: float


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) + eps)
    )


def make_tracer_bases(
    Ct1_bio: np.ndarray,
    Ct2_bio: np.ndarray,
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
    w1, _ = nnls(Gamma_1, Ct1_bio)
    w2, _ = nnls(Gamma_2, Ct2_bio)

    idx1 = np.where(w1 > eps)[0]
    idx2 = np.where(w2 > eps)[0]

    psi1 = gamma_lib_1[idx1]
    psi2 = gamma_lib_2[idx2]

    phi1 = psi1 * decay1[np.newaxis, :]
    phi2 = psi2 * decay2[np.newaxis, :]

    return phi1.T, phi2.T


def simulate_clean_dual_tac(
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

    Ct1_bio = frame_average(t_int, Ct1_int, timegrid.frame_edges)
    Ct2_bio = frame_average(t_int, Ct2_int, timegrid.frame_edges)

    lam1 = np.log(2.0) / tracer1.half_life_min
    lam2 = np.log(2.0) / tracer2.half_life_min

    t_rel1 = t_frames
    t_rel2 = np.maximum(t_frames - delta_min, 0.0)

    decay1 = np.exp(-lam1 * t_rel1)
    decay2 = np.exp(-lam2 * t_rel2)

    Ct1_phys = Ct1_bio * decay1
    Ct2_phys = Ct2_bio * decay2
    y_clean = Ct1_phys + Ct2_phys

    y_meas = add_noise_voxel(
        y_clean,
        timegrid.frame_durs,
        voxel_size_mm=4.0,
        scanner_name="panorama_gs",
        rng=rng,
    )[0]
    print("y_clean has nan:", np.isnan(y_clean).any(), "inf:", np.isinf(y_clean).any())
    print("y_clean min/max:", np.nanmin(y_clean), np.nanmax(y_clean))

    print("y_meas has nan:", np.isnan(y_meas).any(), "inf:", np.isinf(y_meas).any())
    print("y_meas min/max:", np.nanmin(y_meas), np.nanmax(y_meas))
    print("y_meas:", y_meas)

    return DualTacSimulation(
        t_frames=t_frames,
        delta_min=delta_min,
        Cp1=Cp1_int,
        Cp2=Cp2_int,
        Ct1_bio=Ct1_bio,
        Ct2_bio=Ct2_bio,
        Ct1_phys=Ct1_phys,
        Ct2_phys=Ct2_phys,
        y_clean=y_clean,
        y_meas=y_meas,
        decay1=decay1,
        decay2=decay2,
        t_rel1=t_rel1,
        t_rel2=t_rel2,
        lam1=lam1,
        lam2=lam2,
    )


def estimate_biological_curves(
    tracer1_meas: np.ndarray,
    tracer2_meas: np.ndarray,
    lam1: float,
    lam2: float,
    t_rel1: np.ndarray,
    t_rel2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert measured-domain separated curves back to biological-domain TACs.
    """
    Ct1_est_bio = tracer1_meas * np.exp(lam1 * t_rel1)
    Ct2_est_bio = tracer2_meas * np.exp(lam2 * t_rel2)
    return Ct1_est_bio, Ct2_est_bio


def simulate_dual_tac_any_alg(
    timegrid,
    rac,
    fdg,
    protocol,
    gamma_lib_1: np.ndarray,
    Gamma_1: np.ndarray,
    gamma_lib_2: np.ndarray,
    Gamma_2: np.ndarray,
    rng,
    separation_alg: SeparationAlgorithm,
) -> DualTacResult:
    """
    Simulate a noisy dual-tracer TAC and separate it with an arbitrary algorithm.
    """
    sim = simulate_clean_dual_tac(
        timegrid=timegrid,
        tracer1=rac,
        tracer2=fdg,
        protocol=protocol,
        rng=rng,
    )

    Phi_rac, Phi_fdg = make_tracer_bases(
        sim.Ct1_bio,
        sim.Ct2_bio,
        gamma_lib_1,
        Gamma_1,
        gamma_lib_2,
        Gamma_2,
        sim.decay1,
        sim.decay2,
    )

    separation_result: SeparationResult = separation_alg.separate(
        y=sim.y_meas,
        t_frames = sim.t_frames,
        Phi_1= Phi_rac,
        Phi_2= Phi_fdg,
    )


    Ct1_est_bio, Ct2_est_bio = estimate_biological_curves(
        tracer1_meas=separation_result.tracer1_curve,
        tracer2_meas=separation_result.tracer2_curve,
        lam1=sim.lam1,
        lam2=sim.lam2,
        t_rel1=sim.t_rel1,
        t_rel2=sim.t_rel2,
    )

    return DualTacResult(
        t_frames=sim.t_frames,
        Delta=sim.Delta,
        Cp1=sim.Cp1,
        Cp2=sim.Cp2,
        Ct1_bio=sim.Ct1_bio,
        Ct2_bio=sim.Ct2_bio,
        Ct1_phys=sim.Ct1_phys,
        Ct2_phys=sim.Ct2_phys,
        y_clean=sim.y_clean,
        y_meas=sim.y_meas,
        Ct1_est_bio=Ct1_est_bio,
        Ct2_est_bio=Ct2_est_bio,
        ct_1_est_meas=separation_result.tracer1_curve,
        ct_2_est_meas=separation_result.tracer2_curve,
        nrmse_est_1_bio=nrmse(sim.Ct1_bio, Ct1_est_bio),
        nrmse_est_2_bio=nrmse(sim.Ct2_bio, Ct2_est_bio),
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
    sim = simulate_clean_dual_tac(
        timegrid=timegrid,
        rac=rac,
        fdg=fdg,
        protocol=protocol,
        rng=rng,
    )

    Phi_rac, Phi_fdg = make_tracer_bases(
        sim.Ct1_bio,
        sim.Ct2_bio,
        gamma_lib_1,
        Gamma_1,
        gamma_lib_2,
        Gamma_2,
        sim.decay1,
        sim.decay2,
    )

    if use_joint:
        from separation.sequential_nnls import joint_unmix

        rac_est_phys, fdg_est_phys, _ = joint_unmix(
            sim.y_clean,
            Phi_rac,
            Phi_fdg,
            sim.t_frames,
            t_cut=protocol.early_cut(),
        )
    else:
        from separation.sequential_nnls import sequential_unmix_l2

        rac_est_phys, fdg_est_phys = sequential_unmix_l2(
            sim.y_meas,
            Phi_rac,
            Phi_fdg,
            sim.t_frames,
            protocol.early_cut(),
            alpha=0.3,
        )

    Ct1_est_bio, Ct2_est_bio = estimate_biological_curves(
        tracer1_meas=rac_est_phys,
        tracer2_meas=fdg_est_phys,
        lam1=sim.lam1,
        lam2=sim.lam2,
        t_rel1=sim.t_rel1,
        t_rel2=sim.t_rel2,
    )

    return {
        "t_frames": sim.t_frames,
        "Delta": sim.Delta,
        "Cp1": sim.Cp1,
        "Cp2": sim.Cp2,
        "Ct1_bio": sim.Ct1_bio,
        "Ct2_bio": sim.Ct2_bio,
        "Ct1_phys": sim.Ct1_phys,
        "Ct2_phys": sim.Ct2_phys,
        "y_clean": sim.y_clean,
        "y_meas": sim.y_meas,
        "Ct1_est_bio": Ct1_est_bio,
        "Ct2_est_bio": Ct2_est_bio,
    }
# sim/tac_simulator.py
# import numpy as np
# from kinetics.helpers.twotcm import simulate_2tcm
# from core.noise import add_noise_voxel, add_noise_roi
# from utils.frame_average import frame_average
# from scipy.optimize import nnls
# from separation.base import SeparationAlgorithm, SeparationResult

# def make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2, eps=1e-6):
#     w1, _ = nnls(Gamma_1, Ct1_bio)
#     w2, _ = nnls(Gamma_2, Ct2_bio) #first fit the biological curves to the gamma basis

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

#     Ct1_bio = frame_average(t_int, Ct1_int, timegrid.frame_edges)
#     Ct2_bio = frame_average(t_int, Ct2_int, timegrid.frame_edges)

#     lam1 = np.log(2) / rac.half_life_min
#     lam2 = np.log(2) / fdg.half_life_min
#     t_rel1 = t_frames
#     t_rel2 = np.maximum(t_frames - Delta, 0)

#     decay1 = np.exp(-lam1 * t_rel1)
#     decay2 = np.exp(-lam2 * t_rel2)

#     Ct1_phys = Ct1_bio * decay1
#     Ct2_phys = Ct2_bio * decay2
#     y_clean = Ct1_phys + Ct2_phys

#     y_meas =  add_noise_voxel(y_clean, timegrid.frame_durs, voxel_size_mm=4.0,
#                     scanner_name="panorama_gs", rng=None)[0]

#     Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1,gamma_lib_2, Gamma_2, decay1, decay2)
#     context["Phi_1"] = Phi_rac
#     context["Phi_2"] = Phi_fdg
#     context["t_frames"] = context.get("t_frames", t_frames)
#     context["t_cut"] = context.get("t_cut", 10.0)
#     context["alpha"] = context.get("alpha", 1.0)
#     separation_result: SeparationResult = separation_alg.separate(
#         y=y_meas,
#         x=None,
#         t=t_frames,
#         context=context
#     )

#     Ct1_est_bio = separation_result.tracer1_curve * np.exp(+lam1 * t_rel1)
#     Ct2_est_bio = separation_result.tracer2_curve * np.exp(+lam2 * t_rel2)

#     nrmse_est_1_bio = np.sqrt(np.mean((Ct1_bio - Ct1_est_bio)**2)) / (np.max(Ct1_bio) + 1e-6)
#     nrmse_est_2_bio = np.sqrt(np.mean((Ct2_bio - Ct2_est_bio)**2)) / (np.max(Ct2_bio) + 1e-6)

#     return dict(
#         t_frames=t_frames,
#         Delta=Delta,
#         Cp1=Cp1_int,
#         Cp2=Cp2_int,
#         Ct1_bio=Ct1_bio,
#         Ct2_bio=Ct2_bio,
#         Ct1_phys=Ct1_phys,
#         Ct2_phys=Ct2_phys,
#         y_clean=y_clean,
#         y_meas=y_meas,
#         Ct1_est_bio=Ct1_est_bio,
#         Ct2_est_bio=Ct2_est_bio,
#         ct_1_est_meas = separation_result.tracer1_curve,
#         ct_2_est_meas = separation_result.tracer2_curve,
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

#     Ct1_bio = frame_average(t_int, Ct1_int, timegrid.frame_edges)
#     Ct2_bio = frame_average(t_int, Ct2_int, timegrid.frame_edges)

#     lam1 = np.log(2) / rac.half_life_min
#     lam2 = np.log(2) / fdg.half_life_min
#     t_rel1 = t_frames
#     t_rel2 = np.maximum(t_frames - Delta, 0)

#     decay1 = np.exp(-lam1 * t_rel1)
#     decay2 = np.exp(-lam2 * t_rel2)

#     Ct1_phys = Ct1_bio * decay1
#     Ct2_phys = Ct2_bio * decay2
#     y_clean = Ct1_phys + Ct2_phys

#     y_meas =  add_noise_voxel(y_clean, timegrid.frame_durs, voxel_size_mm=4.0,
#                     scanner_name="panorama_gs", rng=None)[0]

#     Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2)
#     # Check the condition number of the basis matrices
#     # cond_rac = np.linalg.cond(Phi_rac)
#     # cond_fdg = np.linalg.cond(Phi_fdg)
#     # print(f"Condition number of PBR basis: {cond_rac:.2e}")
#     # print(f"Condition number of FDG basis: {cond_fdg:.2e}") 

#     if use_joint:
#         from separation.sequential_nnls import joint_unmix
#         rac_est_phys, fdg_est_phys, info = joint_unmix(y_clean, Phi_rac, Phi_fdg, t_frames, t_cut=protocol.early_cut())
#     else:
#         from separation.sequential_nnls import sequential_unmix_l2
#         t_cut = protocol.early_cut()
#         rac_est_phys, fdg_est_phys = sequential_unmix_l2(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut, alpha=0.3)

#     Ct1_est_bio = rac_est_phys * np.exp(+lam1 * t_rel1)
#     Ct2_est_bio = fdg_est_phys * np.exp(+lam2 * t_rel2)

#     return dict(
#         t_frames=t_frames,
#         Delta=Delta,
#         Cp1=Cp1_int,
#         Cp2=Cp2_int,
#         Ct1_bio=Ct1_bio,
#         Ct2_bio=Ct2_bio,
#         Ct1_phys=Ct1_phys,
#         Ct2_phys=Ct2_phys,
#         y_clean=y_clean,
#         y_meas=y_meas,
#         Ct1_est_bio=Ct1_est_bio,
#         Ct2_est_bio=Ct2_est_bio
#     )
