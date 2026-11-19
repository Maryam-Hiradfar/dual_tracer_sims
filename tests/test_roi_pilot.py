"""Synthetic-only ROI pilot tests (not human-derived inputs)."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls
from sim.roi_pilot import (preferred_frame_edges, frame_activity,
                           sample_roi_noise, resolve_rois, run)


def test_preferred_schedule_clips_at_60():
    edges=preferred_frame_edges(60.)
    assert edges[0]==0
    assert edges[-1]==pytest.approx(60.)
    assert np.all(np.diff(edges)>0)
    assert np.diff(edges)[0]==pytest.approx(2/60)
    assert len(edges)>60


def test_component_time_shift_and_decay():
    mid=np.array([.2,.7,1.2,2.,4.])
    signal=np.array([100,120,140,150,160],float)
    edges=np.array([0.,.5,1.,1.5,2.])
    c=frame_activity(mid,signal,edges,1.,109.8)
    assert np.all(c[:2]==0)
    assert c[2]>0


def test_reference_poisson_cv_scaling():
    rng=np.random.default_rng(5)
    c=np.full(50000,10000.)
    out=sample_roi_noise(c,np.full_like(c,1.),rng,cv_60s=.1)
    assert np.std(out)/np.mean(out)==pytest.approx(.1,rel=.04)


def test_roi_pilot_writes_curves_metrics_and_png(tmp_path):
    seconds=np.arange(15,60*62,30)
    start=seconds-15
    end=seconds+15
    fdg=pd.DataFrame(dict(TimeStart=start,TimeEnd=end,TimeMid=seconds,
                          Cortex=4000+8000*(1-np.exp(-seconds/2400)),
                          WhiteMatter=2400+2000*(1-np.exp(-seconds/1500))))
    pbr=pd.DataFrame(dict(TimeStart=start,TimeEnd=end,TimeMid=seconds,
                          Cortex=7000*np.exp(-seconds/2200)+1000,
                          WhiteMatter=5500*np.exp(-seconds/1800)+400))
    a=tmp_path/"fdg.csv"; b=tmp_path/"pbr.csv"
    fdg.to_csv(a,index=False);pbr.to_csv(b,index=False)
    d,s=run(a,b,tmp_path/"out",duration_min=30.,offsets=(0,5),
            noise_levels=(.05,),reps=3)
    assert set(d.roi)=={"Cortex","WhiteMatter"}
    np.testing.assert_allclose(d.mixture_clean_bqml,
                               d.fdg_phys_bqml+d.pbr_phys_bqml)
    before=d[(d.offset_min==5)&(d.TimeEnd<=300)]
    assert (before.fdg_phys_bqml==0).all()
    assert len(s)==4
    for name in ("roi_mixture_tacs.csv","roi_noise_summary.csv",
                 "oracle_template_fit_trials.csv","01_roi_pilot_figure.png",
                 "02_noise_sensitivity.png","pilot_metadata.json"):
        assert (tmp_path/"out"/name).exists()


def test_template_nnls_is_oracle():
    f=np.array([0.,0.,1.,2.,4.])
    p=np.array([4.,3.,2.,1.,.5])
    estimates,_=nnls(np.column_stack([f,p]),f+p)
    np.testing.assert_allclose(estimates,[1,1],rtol=1e-10,atol=1e-10)


def test_mapping_requires_known_rois():
    with pytest.raises(ValueError,match="not matched"):
        resolve_rois(["A"],["B"],requested=["A"])
