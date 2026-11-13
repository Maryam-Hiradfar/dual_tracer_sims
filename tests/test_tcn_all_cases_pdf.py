"""Smoke test full TCN test case report: no test case omitted."""
import numpy as np
from ml.tcn_all_cases_pdf import export_all_test_cases_pdf


def test_all_cases_pdf_contains_all_test_samples(tmp_path):
    n=11
    t=np.linspace(0,20,12)
    reference=np.stack((
        np.tile(5000*(1-np.exp(-t/4)),(n,1)),
        np.tile(3000*np.exp(-t/9)+100,(n,1)),
    ),axis=1)
    prediction=reference*1.03
    clean=reference.sum(axis=1,keepdims=True)
    metadata=[dict(sample_id=k,delta_min=[0,5,10][k%3]) for k in range(n)]
    result=export_all_test_cases_pdf(
        tmp_path/"all_cases.pdf",time_min=t,
        measured=clean,clean=clean,target=reference,prediction=prediction,
        metadata=metadata,cases_per_page=3)
    assert result["test_cases"]==n
    assert result["pages"]==5   # group sizes 4,4,3 -> 2+2+1 
    assert sum(result["offset_groups"].values())==n
    assert (tmp_path/"all_cases.pdf").stat().st_size>1000
