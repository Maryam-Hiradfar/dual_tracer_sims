# kinetics/basis_gamma.py
import numpy as np
from kinetics.twotcm import simulate_2tcm
from tracers import PBR28Tracer, FDGTracer
from tracers.aif import pbr28_aif, fdg_aif


def gamma_shape(t, t0, tau, scale= 1.0):
    x = np.maximum(t - t0, 0)
    return scale * (x / tau) * np.exp(-x / tau)

def build_gamma_library(t, n_t0, n_tau, scale=1.0):
    t0_values  = np.linspace(0, float(t[-1]) - 10, n_t0)
    tau_values = np.linspace(0.3, 40, n_tau)

    lib = []
    for t0 in t0_values:
        for tau in tau_values:
            lib.append(gamma_shape(t, t0, tau, scale=scale))

    lib = np.array(lib)
    return lib, lib.T     # gamma_lib, Gamma



def build_two_tcm_library(t, t_0_vals, tracer_name, scale=1.0, tracer_params = None, param_variations = None):
    t0_values  = t_0_vals if t_0_vals is not None else np.linspace(0, float(t[-1]) - 10, n_tau)
    tau_values = np.linspace(0.3, 40, n_tau)
    k_1 = tracer_params.get("K1", 0.2)
    k_2 = tracer_params.get("k2", 0.2)
    k_3 = tracer_params.get("k3", 0.05)
    k_4 = tracer_params.get("k4", 0.01)
    lib = []
    K1_num_variations = param_variations.get("K1", 5) if param_variations is not None else 1
    k_2_num_variations = param_variations.get("k2", 5) if param_variations is not None else 1
    k_3_num_variations = param_variations.get("k3", 5)
    k_4_num_variations = param_variations.get("k4", 5)
    k_1_variation_range = param_variations.get("K1_range", (0.1, 0.3)) if param_variations is not None else (k_1, k_1)
    k_2_variation_range = param_variations.get("k2_range", (0.1, 0.3)) if param_variations is not None else (k_2, k_2)
    k_3_variation_range = param_variations.get("k3_range", (0.01, 0.1)) if param_variations is not None else (k_3, k_3)
    k_4_variation_range = param_variations.get("k4_range", (0.005, 0.02)) if param_variations is not None else (k_4, k_4)
    k_1_values = np.linspace(k_1_variation_range[0], k_1_variation_range[1], K1_num_variations)
    k_2_values = np.linspace(k_2_variation_range[0], k_2_variation_range[1], k_2_num_variations)
    k_3_values = np.linspace(k_3_variation_range[0], k_3_variation_range[1], k_3_num_variations)
    k_4_values = np.linspace(k_4_variation_range[0], k_4_variation_range[1], k_4_num_variations)

    #tracer_1 input function based on the relevant tracer input function
    if tracer_name == "PBR28":
        aif_func = pbr28_aif
    elif tracer_name == "FDG":
        aif_func = fdg_aif
    else:
        raise ValueError(f"Unsupported tracer name: {tracer_name}")
    

    for t0 in t0_values:
        for k1 in k_1_values:
            for k2 in k_2_values:
                for k3 in k_3_values:
                    for k4 in k_4_values:
                        tracer_params = dict(K1=k1, k2=k2, k3=k3, k4=k4)
                        tac = simulate_2tcm(t, aif_func, t0, tracer_params)
                        lib.append(scale * tac)


    lib  = np.array(lib)    
    return lib, lib.T     # tcm_lib, TCM_Gamma

def build_gamma_library_for_two_tracers(t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1= 2, scale_2=3):
    lib_1, Gamma_1 = build_gamma_library(t, n_t0_1, n_tau_1, scale=scale_1)
    lib_2, Gamma_2 = build_gamma_library(t, n_t0_2, n_tau_2, scale=scale_2)
    return lib_1, Gamma_1, lib_2, Gamma_2
# def build_gamma_library_for_two_auto_scale(t, n_t0_1, n_tau_1, n_t0_2, n_tau_2, scale_1=1.0, scale_2=1.0):
# """
# uses the peak finder to set the average peak of the gamma functions in each library to the provided scale for that tracer, then builds libraries by 
# varying 
# """
def build_gamma_library_for_two_specific_injection_time_points(t, t0_values_1, n_tau_1, t0_values_2, n_tau_2, scale_1= 1, scale_2=1):
    lib_1 = []
    for t0 in t0_values_1:
        tau_values = np.linspace(0.3, 40, n_tau_1)
        for tau in tau_values:
            lib_1.append(gamma_shape(t, t0, tau, scale=scale_1))
    lib_1 = np.array(lib_1)
    Gamma_1 = lib_1.T

    lib_2 = []
    for t0 in t0_values_2:
        tau_values = np.linspace(0.3, 40, n_tau_2)
        for tau in tau_values:
            lib_2.append(gamma_shape(t, t0, tau, scale=scale_2))
    lib_2 = np.array(lib_2)
    Gamma_2 = lib_2.T

    return lib_1, Gamma_1, lib_2, Gamma_2
def build_kinetics_informed_library_for_two_tracers(t, tracer_1, tracer_2,tracer_1_param_variations, tracer_2_param_variations, scale_1= 1.0, scale_2=1.0,
                                                     tracer_1_injection_time = 0.0, tracer_2_injection_time = 0.0, time_variation_range_1 = 10.0,
                                                       time_variation_range_2 = 10.0, num_variations_1 = 5, num_variations_2 = 5):
    #Use the tracer parameters to set the range of t0 and tau values for each tracer, then build libraries by varying those parameters within those ranges
    tracer_1_params = tracer_1.true_params()
    tracer_2_params = tracer_2.true_params()
    tracer_1_t0_range = (max(0, tracer_1_injection_time - time_variation_range_1), tracer_1_injection_time + time_variation_range_1)
    tracer_2_t0_range = (max(0, tracer_2_injection_time - time_variation_range_2), tracer_2_injection_time + time_variation_range_2)
    tracer_1_t0_values = np.linspace(tracer_1_t0_range[0], tracer_1_t0_range[1], num_variations_1)
    tracer_2_t0_values = np.linspace(tracer_2_t0_range[0], tracer_2_t0_range[1], num_variations_2)



    #make libraries using the build_two_tcm_library function, which simulates the 2TCM with the given parameters and uses the resulting TACs as basis functions in the library
    lib_1, Gamma_1 = build_two_tcm_library(t, tracer_1_t0_values, tracer_name=tracer_1.name, scale=scale_1, tracer_params=tracer_1_params, param_variations=tracer_1_param_variations)
    lib_2, Gamma_2 = build_two_tcm_library(t, tracer_2_t0_values, tracer_name=tracer_2.name, scale=scale_2, tracer_params=tracer_2_params, param_variations=tracer_2_param_variations)

    return lib_1, Gamma_1, lib_2, Gamma_2

   

   
   