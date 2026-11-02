"""Draw a presentation diagram of the mixture_30ep TCN (no model changes)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE = "#d9e9f7"
INK = "#23354c"
GREEN = "#78a86c"
GOLD = "#e7bc54"
PURPLE = "#947bb5"


def main():
    fig, ax = plt.subplots(figsize=(18, 12), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 180), ylim=(0, 120))
    ax.axis("off")

    def label(x, y, s, size=13, color=INK, ha="center", **kw):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", **kw)

    def box(x, y, w, h, text="", fc=BLUE, ec=GOLD, size=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5,rounding_size=1.2",
                     facecolor=fc, edgecolor=ec, linewidth=1.7))
        if text:
            label(x+w/2, y+h/2, text, size)

    def arrow(x1, y1, x2, y2, color=INK, lw=1.6):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    shrinkA=2, shrinkB=2))

    def neuron(x, y, activation="ReLU", width=29):
        box(x, y-5.5, width, 11, ec="white")
        for cx, txt in [(x+7, r"$\Sigma$"), (x+width-7, activation)]:
            ax.add_patch(Circle((cx,y), 4.3, facecolor=BLUE,
                                edgecolor=INK, linewidth=1.6))
            label(cx,y,txt,22 if txt == r"$\Sigma$" else 10)
        arrow(x+11.5,y,x+width-11.5,y)

    label(7,114,"Our TCN: one mixture in, two tracer curves out",27,ha="left",weight="bold")
    label(7,109,"Actual prototype architecture • causal sequence-to-sequence separation",14,ha="left")

    label(7,101,"1  THE COMPLETE NETWORK",14,ha="left",weight="bold")
    boxes = [(7,22,"Noisy mixture\n1 × 62"),
             (36,23,"Block 1\ndilation 1\n16 × 62"),
             (66,23,"Block 2\ndilation 2\n16 × 62"),
             (96,23,"Block 3\ndilation 4\n16 × 62"),
             (126,23,"Block 4\ndilation 8\n16 × 62"),
             (156,18,"1×1 Conv\n16 → 2")]
    for i,(x,w,txt) in enumerate(boxes):
        box(x,81,w,15,txt,fc="#eee7f5" if i==0 else BLUE,size=12)
        if i<len(boxes)-1:
            arrow(x+w,88.5,boxes[i+1][0],88.5)
    label(165,76,"FDG, PBR28\n2 × 62",12)
    label(7,73,"Shapes are channels × frames for one scan. All 62 output frames are retained.",12,ha="left")

    # The local calculation resembles the supplied neuron illustration.
    label(7,65,"2  ZOOM IN: ONE CAUSAL CONVOLUTION",14,ha="left",weight="bold")
    label(7,60,"First convolution, kernel size 3, dilation 1",12,ha="left")
    inputs=[(16,50,r"$x_{t-2}$"),(16,38,r"$x_{t-1}$"),(16,26,r"$x_t$")]
    ys=[50,38,26]
    for x,y,txt in inputs:
        label(x,y,txt,20,color=PURPLE)
        for yy in ys:
            arrow(x+6,y,42,yy,color=GREEN,lw=1.1)
    label(29,56,"learned weights",11,color=GREEN)
    for j,y in enumerate(ys):
        neuron(42,y)
        arrow(72,y,79,y)
        label(81,y,rf"$h_{{{j+1},t}}$",17,ha="left")
    label(56,17,"… 16 learned features at frame t",12)
    label(7,10,"Same filters slide across frames; future inputs are excluded.",11,ha="left")
    label(7,5,"Each Σ includes a learned bias. Later convolutions also mix input channels.",10.5,ha="left")

    # Block details occupy a separate, clearly bounded column.
    ax.plot([95,95],[5,67],color="#dbe0e7",lw=1.2)
    label(102,65,"3  INSIDE EACH RESIDUAL BLOCK",14,ha="left",weight="bold")
    box(108,49,54,10,"Causal Conv → ReLU → Dropout",size=11)
    box(108,32,54,10,"Causal Conv → ReLU → Dropout",size=11)
    arrow(135,62,135,59)
    arrow(135,49,135,42)
    arrow(135,32,135,26)
    ax.add_patch(Circle((135,23),3,facecolor="white",edgecolor=INK,lw=1.6))
    label(135,23,"+",20)
    ax.plot([135,169,169],[61,61,23],color=GREEN,lw=1.8)
    arrow(169,23,138,23,color=GREEN)
    label(171,45,"residual\nshortcut",10,color=GREEN)
    arrow(135,20,135,15)
    box(120,7,30,8,"ReLU",size=12)
    label(102,2,"Block 1 shortcut: 1×1 Conv (1 → 16); others: identity.",10,ha="left")

    fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.06)
    fig.text(.5,.025,
             "Kernel = 3  •  Dropout = 0  •  Receptive field = 61 frames  •  Final output is linear (no ReLU)\n"
             "Inputs are normalized; outputs are physical, decayed TACs. Decay correction occurs afterward. Time spacing varies by frame.",
             ha="center",va="center",fontsize=11,color=INK,linespacing=1.7)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT/f"tcn_architecture.{ext}",facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
