"""Create a one-slide preliminary ROI-only PET figure from sim.roi_pilot outputs.

Requires python-pptx. Uses derived PNGs, no participant files in repository.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def make_slide(fig_dir: Path, output: Path, *, title=None):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

    fig_dir=Path(fig_dir)
    big=fig_dir/"01_roi_pilot_figure.png"
    secondary=fig_dir/"02_noise_sensitivity.png"
    if not big.is_file() or not secondary.is_file():
        raise FileNotFoundError("Run python -m sim.roi_pilot first to generate PNG panels")
    pres=Presentation()
    pres.slide_width=Inches(13.333)
    pres.slide_height=Inches(7.5)
    slide=pres.slides.add_slide(pres.slide_layouts[6])

    def textbox(x,y,w,h,text,pt=18,bold=False,color=(25,33,48)):
        shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        frame=shape.text_frame
        frame.word_wrap=True
        p=frame.paragraphs[0]
        p.text=text
        p.font.size=Pt(pt)
        p.font.bold=bold
        p.font.color.rgb=RGBColor(*color)
        return shape
    textbox(.55,.23,12.3,.6,
            title or "FDG/PBR28 ROI-level synthetic mixture feasibility",24,True)
    slide.shapes.add_picture(str(big),Inches(.45),Inches(1.03),width=Inches(9.25))
    slide.shapes.add_picture(str(secondary),Inches(9.55),Inches(2.03),width=Inches(3.5))
    textbox(9.7,1.12,3.2,.7,"Noise sensitivity",16,True)
    textbox(.65,6.77,12.,.55,
            "Preliminary: empirical single-tracer ROI TACs; synthetically delayed/decayed mixture. "
            "Noise illustrative, not calibrated; no blinded tracer separation shown.",11,False,(80,88,99))
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    pres.save(str(output))
    print(f"Saved editable PowerPoint slide to {output}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--title")
    a=p.parse_args()
    make_slide(a.fig_dir,a.out,title=a.title)


if __name__=="__main__":
    main()
