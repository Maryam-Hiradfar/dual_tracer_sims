# # Old code, needs to get updated to work with the refactored code (06/12/2026) - Maryam Hiradfar
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# def plot_nrmse_heatmap(nrmse_matrix, delays, scales, title, tracer_label,
#                        vmin=None, vmax=None, cmap="viridis"):
#     fig, ax = plt.subplots(figsize=(8, 5))

#     im = ax.imshow(
#         nrmse_matrix,
#         aspect="auto",
#         origin="lower",
#         extent=[delays[0], delays[-1], scales[0], scales[-1]],
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#     )

#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(f"{tracer_label} NRMSE (bio)")

#     ax.set_xlabel("FDG injection delay Δ (min)")
#     ax.set_ylabel("PBR:FDG scale ratio")
#     ax.set_title(title)

#     fig.tight_layout()
#     return fig
