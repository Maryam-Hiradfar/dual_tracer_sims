
from sim.config import ExperimentConfig
from sim.experiment_runner import run_single_experiment
from sim.frame_schedules import default_frame_edges
from separation.configs import RecursiveGammaSeparationConfig
from sim.config import TwoTracerBasisConfig, TwoTracerBasisConfig
from kinetics.configs import GammaBasisConfig
from separation.registry import register_all_algorithms
def main() -> None:
    register_all_algorithms()
    cfg = ExperimentConfig(
        experiment_id="test_recursive",
        description="Recursive unmix test",
        frame_edges_min=default_frame_edges(),
        gamma_library = TwoTracerBasisConfig(
            tracer_1_basis=GammaBasisConfig(n_t0=40, n_tau=40),
            tracer_2_basis=GammaBasisConfig(n_t0=40, n_tau=40),
        ),
        algorithm_name="recursive_unmix",
        algorithm_config=RecursiveGammaSeparationConfig(
            alpha_stage_1=1e-3,
            alpha_stage_2=1e-3,
            num_iters=5,
            t_cut = 10.0,
        ),
    )

    summary = run_single_experiment(cfg)
    print(summary)


if __name__ == "__main__":
    main()