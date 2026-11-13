"""Synthetic-only validation of parametric AIF -> 2TCM -> measured frames."""
import json
import numpy as np
import pandas as pd
import pytest

from sim.parametric_tcn_dataset import (
    analytic_aif, dense_frame_means, forward, generate, read_frames,
    tcm_curve,
)


def sample_config():
    aif = {
        "delay_min": 0.03,
        "bolus_amplitude_bqml": 50000.,
        "bolus_peak_min": .35,
        "bolus_shape": 2.,
        "fast_amplitude_bqml": 3000.,
        "fast_lambda_per_min": .6,
        "slow_amplitude_bqml": 1200.,
        "slow_lambda_per_min": .02,
    }
    return dict(
        aif={"FDG": aif, "PBR28": dict(aif, bolus_amplitude_bqml=38000.)},
        kinetics={
            "FDG": {"K1": .146969833218802, "k2": .119575138861569,
                    "k3": .042108367298586, "k4": 0.},
            "PBR28": {"K1": .18, "k2": .25, "k3": .04, "k4": .01},
        },
        vascular_fraction={"FDG": .08, "PBR28": .05},
        reference_doses_mci={"FDG": 4.91, "PBR28": 4.98},
        dose_ranges_mci={"FDG": [4.91, 4.91], "PBR28": [4.98, 4.98]},
        kinetic_variation_fraction=.1,
        aif_variation_fraction=.05,
        fdg_extra_delay_sec=5.78,
    )


def test_analytic_aif_and_tissue_are_finite():
    cfg=sample_config()
    t=np.arange(0,10,1/60)
    cp=analytic_aif(t,cfg["aif"]["FDG"])
    assert cp[0]==0
    assert np.max(cp)>0
    tissue=tcm_curve(cp,1/60,cfg["kinetics"]["FDG"],.08)
    assert np.isfinite(tissue).all()
    assert np.min(tissue)>=0


def test_frames_preserve_native_irregular_boundaries(tmp_path):
    path=tmp_path/"timing.csv"
    pd.DataFrame(dict(TimeStart=[0,2,7,17],TimeEnd=[2,7,17,37])).to_csv(path,index=False)
    frames=read_frames(path,duration_min=30/60)
    np.testing.assert_allclose(frames*60,[[0,2],[2,7],[7,17],[17,30]])


def test_forward_has_zero_fdg_before_injection():
    cfg=sample_config()
    frames=np.array([[0,1],[1,2],[2,3],[3,4],[4,5]],float)
    target,dc,idifs=forward(cfg,frames,3.,4.91,4.98,.0,
                            np.random.default_rng(42))
    assert target.shape==(2,5)
    assert np.all(target[0,:3]==0.)
    assert np.any(target[1,:3]>0.)
    assert np.isfinite(target).all()


def test_generate_tcn_contract_and_original_framing(tmp_path):
    cfg=sample_config()
    params=tmp_path/"params.json";params.write_text(json.dumps(cfg))
    schedule=tmp_path/"timing.csv"
    starts=np.r_[0,np.cumsum([2,5,10,20,30,60,120,180])]
    pd.DataFrame(dict(TimeStart=starts[:-1],TimeEnd=starts[1:])).to_csv(schedule,index=False)
    out=tmp_path/"tcn"
    generate(params,schedule,out,duration_min=float(starts[-1]/60),
             n_samples=30,cv60=.05)
    m=json.loads((out/"dataset_metadata.json").read_text())
    np.testing.assert_allclose(m["frame_durations_min"],np.diff(starts)/60)
    with np.load(out/"train.npz") as d:
        x=d["X_measured_tavg_n1_d1"]
        y=d["Y_phys_tavg_n0_d1"]
        assert x.shape==(24,1,8)
        assert y.shape==(24,2,8)
        np.testing.assert_allclose(
            d["X_clean_tavg_n0_d1"][:,0,:],np.sum(y,axis=1),
            rtol=1e-5,atol=.01)
        assert np.all(np.isfinite(x))
