"""Synthetic Enhanced PET regression test; no human-derived DICOM data."""
import numpy as np
import pytest
pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian
from sim.dicom_dynamic_pet import convert_dicom_directory
from sim.enhanced_pet import frame_info


def seq(**kwargs):
    d=Dataset()
    for k,v in kwargs.items(): setattr(d,k,v)
    return Sequence([d])


def test_per_frame_geometry_and_scaling():
    d=Dataset()
    d.ImageOrientationPatient=[1,0,0,0,1,0]
    d.PixelSpacing=[2,2]
    d.RescaleSlope=2
    d.RescaleIntercept=7
    pf=Dataset()
    pf.PlanePositionSequence=seq(ImagePositionPatient=[0,0,4])
    pf.FrameContentSequence=seq(TemporalPositionIndex=3)
    d.PerFrameFunctionalGroupsSequence=Sequence([pf])
    info=frame_info(d,0)
    assert info["key"]==("temporal_index",3)
    np.testing.assert_allclose(info["pos"],[0,0,4])
    assert info["slope"]==2


def test_3d_multiframe_2_times_converted(tmp_path):
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=pydicom.uid.EnhancedPETImageStorage
    meta.MediaStorageSOPInstanceUID=pydicom.uid.generate_uid()
    folder=tmp_path/"enhanced";folder.mkdir()
    ds=FileDataset(str(folder/"pet.dcm"),{},file_meta=meta,preamble=b"\0"*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID=pydicom.uid.generate_uid()
    ds.Modality="PT"; ds.Units="BQML"
    ds.Rows=4; ds.Columns=5
    ds.NumberOfFrames=6
    ds.SamplesPerPixel=1;ds.PhotometricInterpretation="MONOCHROME2"
    ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
    shared=Dataset()
    shared.PlaneOrientationSequence=seq(ImageOrientationPatient=[1,0,0,0,1,0])
    shared.PixelMeasuresSequence=seq(PixelSpacing=[2,2],SliceThickness=2)
    shared.PixelValueTransformationSequence=seq(RescaleSlope=2,RescaleIntercept=3)
    ds.SharedFunctionalGroupsSequence=Sequence([shared])
    frames=[];pixels=[]
    # Deliberately interleaved time and reversed spatial order.
    for z,t in [(2,1),(1,0),(0,1),(2,0),(0,0),(1,1)]:
        f=Dataset()
        f.PlanePositionSequence=seq(ImagePositionPatient=[0,0,z*2])
        f.FrameContentSequence=seq(TemporalPositionIndex=t+1)
        frames.append(f)
        pixels.append(np.full((4,5),100+t*10+z,dtype="<u2"))
    ds.PerFrameFunctionalGroupsSequence=Sequence(frames)
    ds.PixelData=np.asarray(pixels,dtype="<u2").tobytes()
    ds.save_as(str(folder/"pet.dcm"),enforce_file_format=True)
    timing=tmp_path/"frames.csv"
    timing.write_text("TimeStart,TimeEnd\n0,5\n5,10\n")
    out=tmp_path/"out.nii.gz"
    convert_dicom_directory(folder,out,timing_csv=timing)
    image=nib.load(str(out))
    assert image.shape==(4,5,3,2)
    np.testing.assert_allclose(np.asarray(image.dataobj)[0,0,1,:],[205,225])
