"""Generate editable atlas-label to FDG/PBR28 spreadsheet column mapping."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np
import pandas as pd
from sim.populate_brain_phantom import table


def normalized(s):
    return re.sub(r"[^a-z0-9]+","",str(s).lower())


def template(ids,fdg_cols,pbr_cols,names=None):
    names=names or {}
    def match(label,cols):
        hits=[c for c in cols if normalized(c)==normalized(label)]
        return hits[0] if len(hits)==1 else ""
    return pd.DataFrame([
        dict(label_id=int(i),region_name=names.get(int(i),str(i)),
             fdg_column=match(names.get(int(i),str(i)),fdg_cols),
             pbr_column=match(names.get(int(i),str(i)),pbr_cols))
        for i in ids if i>0])


def main():
    import nibabel as nib
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--fdg-rois",type=Path,required=True)
    p.add_argument("--pbr-rois",type=Path,required=True)
    p.add_argument("--atlas-lut",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    im=nib.load(str(a.atlas))
    if len(im.shape)!=3: raise ValueError("atlas must be 3D")
    labels=np.unique(np.asarray(im.dataobj))
    if not np.allclose(labels,np.rint(labels)): raise ValueError("integer labels required")
    names={}
    if a.atlas_lut:
        d=pd.read_csv(a.atlas_lut)
        if not {"label_id","region_name"}.issubset(d.columns):
            raise ValueError("LUT needs label_id,region_name")
        names={int(row.label_id):str(row.region_name) for row in d.itertuples(index=False)}
    exclude={"TimeStart","TimeEnd","TimeMid","aorta","Cp"}
    fdg=[str(c) for c in table(a.fdg_rois).columns if c not in exclude]
    pbr=[str(c) for c in table(a.pbr_rois).columns if c not in exclude]
    df=template(labels,fdg,pbr,names)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.out,index=False)
    unmatched=((df.fdg_column=="")|(df.pbr_column=="")).sum()
    print(f"Created {a.out} with {len(df)} labels; {unmatched} need manual mapping or exclusion")
    print("FDG TAC columns:",fdg)
    print("PBR28 TAC columns:",pbr)


if __name__=="__main__":
    main()
