#canonicalize channel order, stack arrays, save a clean dataset object
import numpy as np
def canonicalize_sample(sample, spec): 
    """
    force output channels to always mean:
    channel 0 = FDG
    channel 1 = PBR28
    """
    if spec.tracer1_name =="FDG" and spec.tracer2_name == "PBR28":
        y_fdg = sample.y_tracer1_true
        y_pbr = sample.y_tracer2_true
    elif spec.tracer1_name =="PBR28" and spec.tracer2_name=="FDG":
        y_fdg = sample.y_tracer2_true
        y_pbr = sample.y_tracer1_true
    else:
        raise ValueError (
            f"Unexpected tracer order: {spec.tracer1_name}, {spec.tracer2_name}"
        )
    return {
        "x" : sample.y_meas, 
        "y_fdg": y_fdg, 
        "y_pbr28": y_pbr,
        "metadata": {
            **sample.metadata, 
            "tracer1_name": spec.tracer1_name,
            "tracer2_name": spec.tracer2_name, 
            "tracer1_scale": spec.tracer1_scale, 
            "tracer2_scale": spec.tracer2_scale, 
            "delta_min": spec.delta_min,
            "tracer1_first": spec.tracer1_first,
            "seed": spec.seed, 
        },

    }        
def stack_samples(samples, spec):
    canon = [canonicalize_sample(s, sp) for s, sp in zip(samples, spec)]
    X = np.stack([c["x"] for c in canon], axis = 0)   #[N, T]
    Y_fdg = np.stack([c["y_fdg"] for c in canon], axis = 0) #[N,T]
    Y_pbr  = np.stack([c["y_pbr"] for c in canon], axis = 0) #[N,T]
    #neural networks expect (batch, chanels, time), so we need to slice the arrays to also include number of channels
    X = X[:, None, :]
    Y  = np.stack([Y_fdg, Y_pbr], axis = 1)
    
    metadata = [c["metadata"] for c in canon]
    return X, Y, metadata

def save_datasets_npz(path, X, Y, metadata):


def load_dataset_npz(path):