"""Synthetic-only sampled-frame selector tests."""
import numpy as np
from sim.sample_roi_dicom_noise import select_frames


def test_duration_selection_without_duplicate_frames():
    edges=np.array([[0,5],[5,10],[10,40],[40,100],[100,400]],float)
    assert select_frames(edges,(5,30,60,300))==[0,2,3,4]


def test_no_requested_duration_preserves_empty_selection():
    edges=np.array([[0,5],[5,10]],float)
    assert select_frames(edges,())==[]
