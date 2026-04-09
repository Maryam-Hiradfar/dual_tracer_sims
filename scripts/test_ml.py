from sim import sample_generator
import numpy as np 
from core.timegrid import TimeGrid
from tracers.fdg import FDGTracer
from tracers.pbr28 import PBR28Tracer
from core.protocol import Protocol
from sim.sample_generator import DataSetConfig,DatasetGenerator
from tracers.tracer_factory import tracer_factory
import matplotlib.pyplot as plt



FDG_OVERRIDES = {}
PBR_OVERRIDES = {}
# Frame durations in MINUTES
frame_durations_min = np.array(
    [2/60.0] * 20    # 20 × 2 s  = 40 s
    + [5/60.0] * 12    # 12 × 5 s  = 60 s
    + [10/60.0] * 12   # 12 × 10 s = 120 s
    + [20/60.0] * 12   # 12 × 20 s = 240 s
    + [30/60.0] * 8    # 8 × 30 s  = 240 s
    + [1.0]     * 6    # 6 × 60 s  = 360 s
    + [2.0]     * 5    # 5 × 120 s = 600 s
    + [5.0]     * 8    # 8 × 300 s = 2400 s
    + [10.0]    * 4    # 4 × 600 s = 2400 s
)
FRAME_DURATIONS_MIN  = frame_durations_min
FRAME_EDGES = np.concatenate([[0.0], np.cumsum(frame_durations_min)])
frame_mids = 0.5* (FRAME_EDGES[:-1] + FRAME_EDGES[1:])

timegrid = TimeGrid(FRAME_EDGES, internal_dt_min=1/60)

tracer1 = PBR28Tracer(
    name="PBR28",
    half_life_min=20.4, 
    scale = 1.0,
    pbr28_params = (PBR_OVERRIDES if PBR_OVERRIDES else None),
    )
tracer2 = FDGTracer(
    name="FDG", 
    half_life_min=109.8,
    scale = 1.0, 
    feng_params = (FDG_OVERRIDES if FDG_OVERRIDES else None),
        )

# protocol = Protocol(delta_min = 10)
# sample  = sample_generator.SampleGenerator(timegrid,
#                             tracer1, 
#                             tracer2, 
#                             protocol,
#                             rng = None,
# )
# test_sample = sample.generate_sample()
# print(test_sample.metadata)
# plt.figure(figsize=(8,5))
# plt.scatter(frame_mids, test_sample.y_meas)
# plt.xlabel("Time(min)")
# plt.ylabel("Activity")
# plt.legend()
# plt.grid(True)
# plt.show()

#==================================
#      testing the dataset generator
#==================================
cfg = DataSetConfig(
    n_sample = 10,
    delta_choices_min = [0,1,2,3,4,5,10,30], 
    scale_range = (0.8, 1.2),
    randomize_order = True,
)

dataset_gen  = DatasetGenerator(
    timegrid = timegrid, 
    tracer_factory = tracer_factory, 
    config = cfg,
    seed = 48,
    )
samples, specs = dataset_gen.generate_dataset()

for n in range(len(samples)):
    y_meas = samples[n].y_meas
    plt.figure(figsize=(8,5))
    plt.scatter(frame_mids, y_meas)
    plt.xlabel("Time(min)")
    plt.ylabel("Activity")
    plt.legend()
    plt.grid(True)
    plt.show()
