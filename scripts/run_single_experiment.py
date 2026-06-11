#seems to be the most up to date at the moment (June 8, 2026)
from sim.config import ExperimentConfig
from sim.experiment_runner import run_single_experiment
from sim.frame_schedules import default_frame_edges
from separation.configs import RecursiveGammaSeparationConfig, JointGammaSeparationConfig, SequentialGammaSeparationConfig
from sim.config import TwoTracerBasisConfig
from kinetics.configs import GammaBasisConfig, InjectionCenteredGammaBasisConfig, KineticsInformedLibraryConfig
from separation.registry import register_all_algorithms
def main() -> None:
    register_all_algorithms()
    cfg = ExperimentConfig(
        experiment_id="recursive",
        description="recursive unmix test",
        frame_edges_min=default_frame_edges(),
        # gamma_library = TwoTracerBasisConfig(
        #     tracer_1_basis=GammaBasisConfig(n_t0=40, n_tau=40),
        #     tracer_2_basis=GammaBasisConfig(n_t0=40, n_tau=40),
        # ),
        gamma_library = TwoTracerBasisConfig(
            tracer_1_basis = InjectionCenteredGammaBasisConfig(injection_time=0, variation_range = 0, num_t0= 1, n_tau = 40, scale = 1, tau_min = 0.003, tau_max = 10),
            tracer_2_basis= InjectionCenteredGammaBasisConfig(injection_time= 15, variation_range = 120, num_t0= 40, n_tau = 40, scale = 1, tau_min = 0.03, tau_max = 10 ),
        ),
        algorithm_name="recursive_unmix",
        algorithm_config=RecursiveGammaSeparationConfig(
            alpha_stage_1=1e-1,
            alpha_stage_2=1e-1,
            num_iters=10,
            t_cut = 10.0,
        ),
        # algorithm_name= "joint_unmix_gamma",
        # algorithm_config= JointGammaSeparationConfig(
        #     alpha = 0.01,
        #     positive_only= True,
        # ),
        # algorithm_name= "sequential_unmix",
        # algorithm_config = SequentialGammaSeparationConfig(
        #     alpha_stage_1= 0.001,
        #     alpha_stage_2= 0.001,
        #     t_cut= 20,
        #     positive_only= True,
        # ),
    )

    summary = run_single_experiment(cfg)
    print(summary)


if __name__ == "__main__":
    main()