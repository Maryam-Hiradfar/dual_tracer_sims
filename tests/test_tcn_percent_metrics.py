"""Percentage metrics: avoid division by pre-injection zero frames."""
import numpy as np
from ml.evaluate_tcn_prototype import _channel_metrics


def test_percent_error_normalized_by_global_true_rms():
    true=np.array([[[0.,0.,10.,10.],[2.,2.,2.,2.]]])
    pred=np.array([[[0.,0.,11.,11.],[2.,3.,2.,3.]]])
    m=_channel_metrics(pred,true)
    expected_fdg=100*np.sqrt((1+1)/4)/np.sqrt((100+100)/4)
    assert np.isclose(m["FDG"]["percent_rmse"],expected_fdg)
    assert np.isclose(m["FDG"]["percent_mae"],100*.5/np.sqrt(50))
    assert np.isclose(m["PBR28"]["percent_rmse"],100*np.sqrt(.5)/2)
    assert np.isfinite(list(m["FDG"].values())).all()


def test_zero_true_curve_does_not_crash():
    true=np.zeros((2,2,4))
    pred=np.zeros_like(true)
    result=_channel_metrics(pred,true)
    assert result["FDG"]["percent_rmse"]==0
