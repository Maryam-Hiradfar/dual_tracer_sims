# sim/tac_simulator.py
import numpy as np
from kinetics.twotcm import simulate_2tcm
from core.noise import add_noise_voxel, add_noise_roi
from utils.frame_average import frame_average
from scipy.optimize import nnls
from separation.base import SeparationAlgorithm, SeparationResult

def make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2, eps=1e-6):
    w1, _ = nnls(Gamma_1, Ct1_bio)
    w2, _ = nnls(Gamma_2, Ct2_bio) #first fit the biological curves to the gamma basis

    idx1 = np.where(w1 > eps)[0] #only keep the basis functions that contribute significantly, this step is only done in simulation to speed things up, i
    #in real data we would not know this and would have to use the full basis
    idx2 = np.where(w2 > eps)[0]

    psi1 = gamma_lib_1[idx1] 
    psi2 = gamma_lib_2[idx2]

    phi1 = psi1 * decay1[np.newaxis, :] #gnerate the basis functions scaled by the decay
    phi2 = psi2 * decay2[np.newaxis, :]

    return phi1.T, phi2.T
def simulate_dual_tac_any_alg(timegrid, rac, fdg, protocol, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, rng, separation_alg: SeparationAlgorithm, context = None):
    t_int = timegrid.t_internal
    t_frames = timegrid.frame_mids
    Delta = protocol.delta_min

    Cp1_int = rac.aif(t_int)
    Cp2_int = fdg.aif(t_int, delta_min=Delta)

    Ct1_int, _, _ = simulate_2tcm(**rac.true_params(), Cp=Cp1_int, t=t_int)
    Ct2_int, _, _ = simulate_2tcm(**fdg.true_params(), Cp=Cp2_int, t=t_int)

    Ct1_bio = frame_average(t_int, Ct1_int, timegrid.frame_edges)
    Ct2_bio = frame_average(t_int, Ct2_int, timegrid.frame_edges)

    lam1 = np.log(2) / rac.half_life_min
    lam2 = np.log(2) / fdg.half_life_min
    t_rel1 = t_frames
    t_rel2 = np.maximum(t_frames - Delta, 0)

    decay1 = np.exp(-lam1 * t_rel1)
    decay2 = np.exp(-lam2 * t_rel2)

    Ct1_phys = Ct1_bio * decay1
    Ct2_phys = Ct2_bio * decay2
    y_clean = Ct1_phys + Ct2_phys

    y_meas =  add_noise_voxel(y_clean, timegrid.frame_durs, voxel_size_mm=4.0,
                    scanner_name="panorama_gs", rng=None)[0]

    Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1,gamma_lib_2, Gamma_2, decay1, decay2)
    context["Phi_1"] = Phi_rac
    context["Phi_2"] = Phi_fdg
    context["t_frames"] = context.get("t_frames", t_frames)
    context["t_cut"] = context.get("t_cut", 10.0)
    context["alpha"] = context.get("alpha", 1.0)
    separation_result: SeparationResult = separation_alg.separate(
        y=y_meas,
        x=None,
        t=t_frames,
        context=context
    )

    Ct1_est_bio = separation_result.tracer1_curve * np.exp(+lam1 * t_rel1)
    Ct2_est_bio = separation_result.tracer2_curve * np.exp(+lam2 * t_rel2)

    nrmse_est_1_bio = np.sqrt(np.mean((Ct1_bio - Ct1_est_bio)**2)) / (np.max(Ct1_bio) + 1e-6)
    nrmse_est_2_bio = np.sqrt(np.mean((Ct2_bio - Ct2_est_bio)**2)) / (np.max(Ct2_bio) + 1e-6)

    return dict(
        t_frames=t_frames,
        Delta=Delta,
        Cp1=Cp1_int,
        Cp2=Cp2_int,
        Ct1_bio=Ct1_bio,
        Ct2_bio=Ct2_bio,
        Ct1_phys=Ct1_phys,
        Ct2_phys=Ct2_phys,
        y_clean=y_clean,
        y_meas=y_meas,
        Ct1_est_bio=Ct1_est_bio,
        Ct2_est_bio=Ct2_est_bio,
        ct_1_est_meas = separation_result.tracer1_curve,
        ct_2_est_meas = separation_result.tracer2_curve,
        nrmse_est_1_bio=nrmse_est_1_bio,
        nrmse_est_2_bio=nrmse_est_2_bio
    )


def simulate_dual_tac(timegrid, rac, fdg, protocol, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, rng, use_joint=False):
    t_int = timegrid.t_internal
    t_frames = timegrid.frame_mids
    Delta = protocol.delta_min

    Cp1_int = rac.aif(t_int)
    Cp2_int = fdg.aif(t_int, delta_min=Delta)

    Ct1_int, _, _ = simulate_2tcm(**rac.true_params(), Cp=Cp1_int, t=t_int)
    Ct2_int, _, _ = simulate_2tcm(**fdg.true_params(), Cp=Cp2_int, t=t_int)

    Ct1_bio = frame_average(t_int, Ct1_int, timegrid.frame_edges)
    Ct2_bio = frame_average(t_int, Ct2_int, timegrid.frame_edges)

    lam1 = np.log(2) / rac.half_life_min
    lam2 = np.log(2) / fdg.half_life_min
    t_rel1 = t_frames
    t_rel2 = np.maximum(t_frames - Delta, 0)

    decay1 = np.exp(-lam1 * t_rel1)
    decay2 = np.exp(-lam2 * t_rel2)

    Ct1_phys = Ct1_bio * decay1
    Ct2_phys = Ct2_bio * decay2
    y_clean = Ct1_phys + Ct2_phys

    y_meas =  add_noise_voxel(y_clean, timegrid.frame_durs, voxel_size_mm=4.0,
                    scanner_name="panorama_gs", rng=None)[0]

    Phi_rac, Phi_fdg = make_tracer_bases(Ct1_bio, Ct2_bio, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, decay1, decay2)
    # Check the condition number of the basis matrices
    # cond_rac = np.linalg.cond(Phi_rac)
    # cond_fdg = np.linalg.cond(Phi_fdg)
    # print(f"Condition number of PBR basis: {cond_rac:.2e}")
    # print(f"Condition number of FDG basis: {cond_fdg:.2e}") 

    if use_joint:
        from separation.sequential_nnls import joint_unmix
        rac_est_phys, fdg_est_phys, info = joint_unmix(y_clean, Phi_rac, Phi_fdg, t_frames, t_cut=protocol.early_cut())
    else:
        from separation.sequential_nnls import sequential_unmix_l2
        t_cut = protocol.early_cut()
        rac_est_phys, fdg_est_phys = sequential_unmix_l2(y_meas, Phi_rac, Phi_fdg, t_frames, t_cut, alpha=0.3)

    Ct1_est_bio = rac_est_phys * np.exp(+lam1 * t_rel1)
    Ct2_est_bio = fdg_est_phys * np.exp(+lam2 * t_rel2)

    return dict(
        t_frames=t_frames,
        Delta=Delta,
        Cp1=Cp1_int,
        Cp2=Cp2_int,
        Ct1_bio=Ct1_bio,
        Ct2_bio=Ct2_bio,
        Ct1_phys=Ct1_phys,
        Ct2_phys=Ct2_phys,
        y_clean=y_clean,
        y_meas=y_meas,
        Ct1_est_bio=Ct1_est_bio,
        Ct2_est_bio=Ct2_est_bio
    )
