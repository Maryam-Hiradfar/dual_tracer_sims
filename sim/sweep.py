# sim/sweep.py
import numpy as np
from sim.tac_simulator import simulate_dual_tac_any_alg
from separation.base import SeparationAlgorithm

def nrmse(true, est):
    return np.sqrt(np.mean((true - est)**2)) / (np.max(true) + 1e-6)



def sweep_delays(delays, timegrid, rac, fdg, gamma_lib_1, Gamma_1, gamma_lib_2, Gamma_2, rng, separation_alg = SeparationAlgorithm, context = None):
    out = []
    for D in delays:
        from core.protocol import Protocol
        protocol = Protocol(delta_min=D)

        sim = simulate_dual_tac_any_alg(
            timegrid=timegrid,
            rac=rac,
            fdg=fdg,
            protocol=protocol,
            gamma_lib_1=gamma_lib_1,
            Gamma_1=Gamma_1,
            gamma_lib_2= gamma_lib_2,
            Gamma_2=Gamma_2,
            rng=rng,
            separation_alg=separation_alg,
            context=context
        )

        out.append(dict(
            Delta=D,
            nrmse_est_1_bio=nrmse(sim["Ct1_bio"], sim["Ct1_est_bio"]),
            nrmse_est_2_bio=nrmse(sim["Ct2_bio"], sim["Ct2_est_bio"]),
            nrmse_est_1_meas=nrmse(sim["Ct1_phys"], sim["ct_1_est_meas"]),
            nrmse_est_2_meas=nrmse(sim["Ct2_phys"], sim["ct_2_est_meas"]),
            sim=sim
        ))

    return out
