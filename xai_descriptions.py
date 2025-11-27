import numpy as np

def describe_gradcam(heatmap: np.ndarray) -> str:
    """
    heatmap: numpy array (H, W), values in [0, 1]
    """
    mean_activation = float(heatmap.mean())
    max_activation = float(heatmap.max())

    h, w = heatmap.shape
    upper = heatmap[:h//2, :].mean()
    lower = heatmap[h//2:, :].mean()
    left = heatmap[:, :w//2].mean()
    right = heatmap[:, w//2:].mean()

    vertical = "upper" if upper > lower else "lower"
    horizontal = "left" if left > right else "right"

    return (
        f"Grad-CAM shows activation with mean intensity {mean_activation:.3f} "
        f"and peak intensity {max_activation:.3f}. "
        f"The strongest focus appears in the {vertical} part of the face, "
        f"biased toward the {horizontal} side."
    )


# def describe_lime(mask: np.ndarray) -> str:
#     """
#     mask: (H, W) binary or integer mask from LIME
#     """
#     coverage = float((mask > 0).mean())
#
#     return (
#         f"LIME highlights approximately {coverage * 100:.1f}% of the facial area. "
#         "These highlighted superpixels represent regions that most influenced "
#         "the model's prediction."
#     )


def describe_lime(mask: np.ndarray, contribution: np.ndarray = None) -> str:
    coverage = float((mask > 0).mean())  # % of image highlighted

    desc = f"LIME highlights {coverage*100:.1f}% of the facial area as influential. "

    if contribution is not None:
        pos = float((contribution > 0).sum()) / contribution.size * 100
        neg = float((contribution < 0).sum()) / contribution.size * 100
        desc += f"Of these highlighted regions, {pos:.1f}% positively contributed to predicting FAKE, " \
                f"and {neg:.1f}% negatively contributed."

    return desc

