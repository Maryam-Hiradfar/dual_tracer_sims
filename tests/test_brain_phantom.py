"""Synthetic mathematical tests; no participant data."""
import csv
import numpy as np
import pandas as pd
import pytest
from sim.estimate_pet_noise import estimate_frame,load_timing_csv
from sim.populate_brain_phantom import concentration,physical_frames,make_curves,read_edges
from sim.make_brain_roi_mapping import template


def test_fdg_zero_before_injection_and_summation():
    midpoint=np.array([.25,.75,1.25])
    activity=np.array([100.,150.,200.])
    t=np.array([0.,.5,1.,1.5])
    assert np.all(concentration(t,midpoint,activity,1.)[:3]==0)
    edges=np.array([0.,.5,1.,1.5])
    assert np.all(physical_frames(midpoint,activity,edges,1.,109.8)[:2]==0)
    cols=dict(TimeMid=[15,45,75,105],TimeStart=[0,30,60,90],TimeEnd=[30,60,90,120])
    fdg=pd.DataFrame({**cols,"Cortex":[100,120,150,170]})
    pbr=pd.DataFrame({**cols,"Cortex":[250,230,200,170]})
    m=pd.DataFrame([dict(label_id=2,region_name="Cortex",fdg_column="Cortex",pbr_column="Cortex")])
    d=make_curves(fdg,pbr,m,np.array([0,.5,1,1.5,2]),.5)
    np.testing.assert_allclose(d.mixture_clean_bqml,d.fdg_phys_bqml+d.pbr_phys_bqml)
    assert d.fdg_phys_bqml.iloc[0]==0


def test_frame_edges_and_mapping(tmp_path):
    path=tmp_path/"timing.csv"
    path.write_text("TimeStart,TimeEnd\n0,5\n5,10\n")
    np.testing.assert_allclose(load_timing_csv(path,2),[[0,5],[5,10]])
    d=pd.DataFrame(dict(TimeStart=[0,5],TimeEnd=[5,10],TimeMid=[2.5,7.5]))
    np.testing.assert_allclose(read_edges(path,d,10/60),[0,5/60,10/60])
    m=template(np.array([0,2,3]),["Cortex","White Matter"],["Cortex"],
               {2:"Cortex",3:"White Matter"})
    assert m.label_id.tolist()==[2,3]
    assert m.pbr_column.iloc[0]=="Cortex"
    assert m.pbr_column.iloc[1]==""


def test_split_half_noise_recovers_known_sigma():
    rng=np.random.default_rng(4)
    sigma_full=120.
    ha=10000+rng.normal(0,sigma_full*np.sqrt(2),(24,24,24))
    hb=10000+rng.normal(0,sigma_full*np.sqrt(2),(24,24,24))
    e=estimate_frame((ha+hb)/2,np.ones(ha.shape,bool),half_a=ha,half_b=hb)
    assert e["sigma_bqml"]==pytest.approx(sigma_full,rel=.09)


def test_spatial_residual_roughly_suppresses_gradient():
    rng=np.random.default_rng(10)
    frame=5000+np.arange(30.)[:,None,None]*100+np.zeros((30,30,30))
    frame=frame+rng.normal(0,100,frame.shape)
    e=estimate_frame(frame,np.ones(frame.shape,bool),erode_vox=3)
    assert 60<e["sigma_bqml"]<180
