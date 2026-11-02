"""Generate and validate a tiny three-sample simulated dataset."""

import numpy as np

from core.timegrid import TimeGrid
from sim.dataset_builder import stack_samples
from sim.frame_schedules import default_frame_edges
from sim.sample_generator import DataSetConfig, DatasetGenerator
from tracers.tracer_factory import tracer_factory


def main() -> None:
    timegrid = TimeGrid(
        frame_edges=default_frame_edges(),
        internal_dt_min=1 / 60,
    )
    config = DataSetConfig(
        n_sample=3,
        delta_choices_min=[0.0, 5.0, 10.0],
        scale_range=(0.8, 1.2),
        first_tracer_name="PBR28",
        second_tracer_name="FDG",
        randomize_order=False,
    )
    generator = DatasetGenerator(
        timegrid=timegrid,
        tracer_factory=tracer_factory,
        config=config,
        seed=48,
    )

    samples, specs = generator.generate_dataset()
    (
        X_measured_tavg_n1_d1,
        X_clean_tavg_n0_d1,
        Y_phys_tavg_n0_d1,
        Y_bio_tavg_n0_d0,
        expected_counts,
        metadata,
    ) = stack_samples(samples, specs)

    assert X_measured_tavg_n1_d1.shape == (config.n_sample, 1, timegrid.n_frames)
    assert X_clean_tavg_n0_d1.shape == (config.n_sample, 1, timegrid.n_frames)
    assert Y_phys_tavg_n0_d1.shape == (config.n_sample, 2, timegrid.n_frames)
    assert Y_bio_tavg_n0_d0.shape == (config.n_sample, 2, timegrid.n_frames)
    assert expected_counts.shape == (3, timegrid.n_frames)
    np.testing.assert_allclose(
        X_clean_tavg_n0_d1[:, 0, :],
        Y_phys_tavg_n0_d1[:, 0, :] + Y_phys_tavg_n0_d1[:, 1, :],
        rtol=1e-6,
        atol=1e-6,
    )

    print("Three-sample dataset generation passed.")
    print(f"X_measured_tavg_n1_d1:     {X_measured_tavg_n1_d1.shape} {X_measured_tavg_n1_d1.dtype}")
    print(f"X_clean_tavg_n0_d1:        {X_clean_tavg_n0_d1.shape} {X_clean_tavg_n0_d1.dtype}")
    print(f"Y_phys_tavg_n0_d1:         {Y_phys_tavg_n0_d1.shape} {Y_phys_tavg_n0_d1.dtype} (FDG, PBR28)")
    print(f"Y_bio_tavg_n0_d0:          {Y_bio_tavg_n0_d0.shape} {Y_bio_tavg_n0_d0.dtype} (FDG, PBR28)")
    print(f"expected_counts:{expected_counts.shape} {expected_counts.dtype}")
    for index, item in enumerate(metadata):
        print(
            f"sample {index}: {item['tracer1_name']} then "
            f"{item['tracer2_name']}, delay={item['delta_min']} min"
        )


if __name__ == "__main__":
    main()
