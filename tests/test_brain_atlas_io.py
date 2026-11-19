"""Synthetic mask-directory import tests."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from sim.brain_atlas_io import load_atlas


def mask(path, arr, affine=None):
    nib.save(nib.Nifti1Image(arr.astype("float32"),
                            np.eye(4) if affine is None else affine), str(path))


def test_directory_masks_get_names_and_labels(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    a=np.zeros((8,8,8)); a[1:3,1:3,1:3]=1
    b=np.zeros((8,8,8)); b[4:6,4:6,4:6]=1
    mask(masks/"GrayMatter.nii.gz",a)
    mask(masks/"WhiteMatter.nii.gz",b)
    im,names=load_atlas(masks,output_dir=tmp_path/"derived")
    labels=np.asarray(im.dataobj)
    assert names == {1:"GrayMatter",2:"WhiteMatter"}
    assert labels[1,1,1] == 1
    assert labels[4,4,4] == 2
    assert (tmp_path/"derived"/"combined_roi_labels.csv").exists()
    assert (tmp_path/"derived"/"combined_roi_labels.nii.gz").exists()


def test_overlapping_masks_rejected(tmp_path):
    a=np.zeros((8,8,8)); a[1:4,1:4,1:4]=1
    mask(tmp_path/"a.nii.gz",a)
    mask(tmp_path/"b.nii.gz",a)
    with pytest.raises(ValueError,match="Overlapping"):
        load_atlas(tmp_path)


def test_misalignment_rejected_by_default(tmp_path):
    a=np.zeros((8,8,8)); a[1:3,1:3,1:3]=1
    mask(tmp_path/"a.nii.gz",a)
    affine=np.diag([2,2,2,1])
    mask(tmp_path/"b.nii.gz",a,affine)
    with pytest.raises(ValueError,match="grid differs"):
        load_atlas(tmp_path)
