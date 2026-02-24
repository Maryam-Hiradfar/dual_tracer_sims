# sim/sweep.py
import numpy as np
from sim.tac_simulator import simulate_dual_tac_any_alg
from separation.base import SeparationAlgorithm

def nrmse(true, est):
    return np.sqrt(np.mean((true - est)**2)) / (np.max(true) + 1e-6)



def sweep_delays(delays, timegrid, rac, fdg, gamma_lib, Gamma, rng, separation_alg = SeparationAlgorithm, context = None):
    out = []
    for D in delays:
        from core.protocol import Protocol
        protocol = Protocol(delta_min=D)

        sim = simulate_dual_tac_any_alg(
            timegrid=timegrid,
            rac=rac,
            fdg=fdg,
            protocol=protocol,
            gamma_lib=gamma_lib,
            Gamma=Gamma,
            rng=rng,
            separation_alg=separation_alg,
            context=context
        )

        out.append(dict(
            Delta=D,
            nrmse_est_1_bio=nrmse(sim["Ct1_bio"], sim["Ct1_est_bio"]),
            nrmse_est_2_bio=nrmse(sim["Ct2_bio"], sim["Ct2_est_bio"]),
            sim=sim
        ))

    return out
