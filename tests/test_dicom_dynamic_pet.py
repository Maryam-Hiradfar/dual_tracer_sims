"""Synthetic-only tests for classic single-frame dynamic PET DICOM conversion."""
from pathlib import Path
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from sim.dicom_dynamic_pet import scan_series, convert_dicom_directory


def synthetic_dicom(path, t, z):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.PositronEmissionTomographyImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "PT"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.99"
    ds.FrameReferenceTime = t*1000
    ds.Units = "BQML"
    ds.Rows = 4
    ds.Columns = 5
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    ds.ImagePositionPatient = [0,0,float(z)*2]
    ds.PixelSpacing = [2,2]
    ds.SliceThickness = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 3
    ds.RescaleIntercept = 5
    ds.PixelData = np.full((4,5), 100+t*10+z, dtype="<u2").tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_group_scale_orientation_and_time(tmp_path):
    d = tmp_path/"dicom"
    d.mkdir()
    # Jumbled directory ordering must not affect slice/frame order.
    for t in (2,0,1):
        for z in (2,0,1):
            synthetic_dicom(d/f"obscure_{3-t}_{2-z}.dcm", t, z)
    groups, aff, shape, keys = scan_series(d)
    assert shape == (4,5,3)
    assert [k[1] for k in keys] == [0,1000,2000]
    assert np.allclose(aff[:3,0],[-2,0,0])
    assert np.allclose(aff[:3,1],[0,-2,0])
    assert np.allclose(aff[:3,2],[0,0,2])
    timing=tmp_path/"timing.csv"
    timing.write_text("TimeStart,TimeEnd\n0,1\n1,2\n2,3\n")
    out=tmp_path/"converted.nii.gz"
    convert_dicom_directory(d,out,timing_csv=timing)
    im=nib.load(str(out))
    assert im.shape == (4,5,3,3)
    np.testing.assert_allclose(im.dataobj[0,0,1,:],[308,338,368])


def test_reject_unverified_units(tmp_path):
    d=tmp_path/"dicom";d.mkdir()
    for t in (0,1):
        for z in (0,1):
            file=d/f"x_{t}_{z}.dcm"
            synthetic_dicom(file,t,z)
            ds=pydicom.dcmread(str(file))
            ds.Units="CNTS"
            ds.save_as(str(file),enforce_file_format=True)
    with pytest.raises(ValueError,match="not BQML"):
        scan_series(d)
