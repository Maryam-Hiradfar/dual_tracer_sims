"""Figure regression tests for model-only TCN visualization."""
import json
import numpy as np
from ml.tcn_diagnostic_plots import render_tcn_diagnostics


def test_all_diagnostic_figures_with_external_legends(tmp_path):
    rng=np.random.default_rng(5)
    t=np.linspace(0,60,40)
    n=9
    y=np.stack([
        np.tile(2500+4000*(1-np.exp(-t/14)),(n,1)),
        np.tile(6000*np.exp(-t/19)+350,(n,1)),
    ],axis=1)
    pred=y*(1+rng.normal(0,.03,size=y.shape))
    clean=y.sum(axis=1,keepdims=True)
    measured=clean+rng.normal(0,100,size=clean.shape)
    metadata=[dict(delta_min=float([0,5,10][i%3]),sample_id=i) for i in range(n)]
    hist=tmp_path/"history.json"
    hist.write_text(json.dumps([
        {"epoch":i,"train":{"loss":.1/i},"val":{"loss":.11/i}}
        for i in range(1,6)
    ]))
    imgs=render_tcn_diagnostics(t,measured,clean,y,pred,metadata,
        {"units":{"TAC":"Bq/mL"},"selected_rois":["aorta"]},
        [0,1,2],tmp_path/"plots",training_history=hist)
    assert len(imgs)==10
    assert (tmp_path/"plots"/"percent_error_by_offset.csv").exists()
    for p in imgs.values():
        assert __import__("pathlib").Path(p).stat().st_size > 1000
