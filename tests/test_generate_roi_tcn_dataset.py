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


def test_aorta_only_inputs_generate_tcn_dataset(tmp_path):
    t=np.arange(15,60*62,30,dtype=float)
    basic=dict(TimeStart=t-15,TimeEnd=t+15,TimeMid=t)
    fdg=pd.DataFrame(dict(**basic,aorta=10000*np.exp(-t/1000)+2500))
    pbr=pd.DataFrame(dict(**basic,aorta=12000*np.exp(-t/650)+1000))
    f=tmp_path/"FDG Aorta.csv";p=tmp_path/"Aorta PBR.csv"
    fdg.to_csv(f,index=False);pbr.to_csv(p,index=False)
    out=tmp_path/"tcn"
    build(f,p,out,duration_min=30,offsets=(0,5,10),n_samples=30,cv60=.05)
    import json
    assert json.loads((out/"dataset_metadata.json").read_text())["selected_rois"]==["aorta"]
    with np.load(out/"test.npz") as d:
        assert d["Y_phys_tavg_n0_d1"].shape[1]==2


def test_measured_framing_retains_actual_pbr_edges_and_scales(tmp_path):
    import json
    from sim.generate_roi_tcn_dataset import measured_frame_edges
    pbr = pd.DataFrame({"TimeStart":[0,10,30,60], "TimeEnd":[10,30,60,120]})
    edges = measured_frame_edges(pbr, 2.)
    np.testing.assert_allclose(edges, [0,10/60,30/60,1,2])
    # If duration cuts a measured frame, use that partial frame endpoint.
    np.testing.assert_allclose(measured_frame_edges(pbr,1.5),
                               [0,10/60,30/60,1,1.5])
    t=np.arange(15,60*62,30,dtype=float)
    base={"TimeStart":t-15,"TimeEnd":t+15,"TimeMid":t}
    fdg=pd.DataFrame({**base,"aorta":4000+1000*(1-np.exp(-t/300))})
    pbr=pd.DataFrame({**base,"aorta":6000*np.exp(-t/2400)+500})
    a=tmp_path/"fdg.csv";b=tmp_path/"pbr.csv";out=tmp_path/"dataset"
    fdg.to_csv(a,index=False);pbr.to_csv(b,index=False)
    build(a,b,out,duration_min=30,offsets=(0,),n_samples=20,
          cv60=.05,frame_schedule="measured",dose_scale=(1.,1.))
    md=json.loads((out/"dataset_metadata.json").read_text())
    assert md["frame_schedule_source"]=="PBR28 TimeStart/TimeEnd"
    assert md["fdg_scaling_range"]==[1.,1.]
    assert md["pbr28_scaling_range"]==[1.,1.]
    np.testing.assert_allclose(np.diff(md["frame_edges_min"]),.5)
    for split in ("train","val","test"):
        metadata=json.loads((out/f"{split}_metadata.json").read_text())
        assert all(x["fdg_scale"]==1 and x["pbr28_scale"]==1 for x in metadata)
