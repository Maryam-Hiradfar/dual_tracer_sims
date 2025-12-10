import matplotlib.pyplot as plt

def plot_nrmse_heatmap(nrmse_matrix, delays, scales, title, tracer_label):
    plt.figure(figsize=(8, 5))

    im = plt.imshow(
        nrmse_matrix,
        aspect="auto",
        origin="lower",
        extent=[delays[0], delays[-1], scales[0], scales[-1]],
    )
    plt.colorbar(im, label=f"{tracer_label} NRMSE (bio)")
    plt.xlabel("FDG injection delay Δ (min)")
    plt.ylabel("PBR:FDG scale ratio")
    plt.title(title)
    plt.tight_layout()
