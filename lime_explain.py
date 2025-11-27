import numpy as np
import torch
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from inference import transform, DEVICE

def lime_predict(images, model):
    """
    images: list of RGB uint8 images (H, W, 3)
    returns: (N, 2) class probabilities [FAKE, REAL]
    """

    batch = []
    for img in images:
        img_resized = cv2.resize(img, (299, 299))
        tensor = transform(img_resized).unsqueeze(0)
        batch.append(tensor)

    batch = torch.cat(batch, dim=0).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs_real = torch.sigmoid(logits).cpu().numpy()

    # 🔴 CRITICAL FIX:
    probs_real = probs_real.reshape(-1, 1)  # (N, 1)
    probs_fake = 1.0 - probs_real

    return np.concatenate([probs_fake, probs_real], axis=1)


# def generate_lime(image, face, model):
#     """
#     image: original BGR image
#     face: cropped 299x299 RGB face
#     """
#
#     explainer = lime_image.LimeImageExplainer()
#
#     explanation = explainer.explain_instance(
#         face,
#         lambda imgs: lime_predict(imgs, model),
#         top_labels=1,
#         hide_color=0,
#         num_samples=2000
#     )
#
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=6,
#         hide_rest=False
#     )
#
#     lime_vis = mark_boundaries(temp / 255.0, mask)
#     lime_vis = (lime_vis * 255).astype(np.uint8)
#
#     return lime_vis




def generate_lime(image, face, model):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    # face: 299x299 RGB
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        face,
        classifier_fn=lambda imgs: lime_predict(imgs, model),
        top_labels=2,
        hide_color=0,
        num_samples=2000
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp / 255.0, mask)

    # Optional: include contributions per superpixel
    contribution = np.zeros_like(mask, dtype=float)
    for superpixel, weight in explanation.local_exp[top_label]:
        contribution[mask == superpixel] = weight

    return lime_img, mask, contribution

