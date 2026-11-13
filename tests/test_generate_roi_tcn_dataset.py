"""Synthetic ROI->existing TCN contract tests."""
import numpy as np
import pandas as pd
from sim.generate_roi_tcn_dataset import build, calibrate_reference_cv


def test_noise_sample_normalization(tmp_path):
    p=tmp_path/"noise.csv"
    pd.DataFrame(dict(cv=[.2,.1],mean_bqml=[10000,10000],
        TimeStart=[0,60],TimeEnd=[60,300])).to_csv(p,index=False)
    cv,meta=calibrate_reference_cv(p,4)
    assert cv>0 and meta["effective_independent_roi_voxels"]==4


def test_tcn_ready_split_shapes(tmp_path):
    t=np.arange(15,60*62,30,dtype=float)
    edges=t-15
    ends=t+15
    base=dict(TimeStart=edges,TimeEnd=ends,TimeMid=t)
    fdg=pd.DataFrame(dict(**base,Cortex=4000+2000*(1-np.exp(-t/300)),
                          WM=2500+1500*(1-np.exp(-t/400))))
    pbr=pd.DataFrame(dict(**base,Cortex=6000*np.exp(-t/2400)+1000,
                          WM=4000*np.exp(-t/2000)+300))
    f=tmp_path/"fdg.csv";p=tmp_path/"pbr.csv"
    fdg.to_csv(f,index=False);pbr.to_csv(p,index=False)
    out=tmp_path/"data"
    build(f,p,out,duration_min=30,offsets=(0,5,10),n_samples=30,
          cv60=.05,seed=12)
    with np.load(out/"train.npz") as data:
        x=data["X_measured_tavg_n1_d1"]
        y=data["Y_phys_tavg_n0_d1"]
        clean=data["X_clean_tavg_n0_d1"]
        assert x.shape[0]==24
        assert x.shape[1]==1
        assert y.shape==(24,2,x.shape[-1])
        np.testing.assert_allclose(y.sum(axis=1,keepdims=True),clean,atol=1e-3)
        assert np.isfinite(x).all()
