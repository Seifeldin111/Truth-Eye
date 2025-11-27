import torch
import torchvision.transforms as T
from model import Net
from face_utils import robust_face_detection
from face_utils import robust_face_detection_from_array

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def load_model():
    model = Net().to(DEVICE)
    model.load_state_dict(
        torch.load("final_model.pt", map_location=DEVICE)
    )
    model.eval()
    return model


@torch.no_grad()
def predict(image, model):
    # image is numpy BGR
    face = robust_face_detection_from_array(image)

    if face is None:
        return None, None, None  # 🟢 return face too

    tensor = transform(face).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

    prob_real = prob
    prob_fake = 1 - prob_real

    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = max(prob_real, prob_fake)

    return label, confidence, face



from grad_cam import GradCAM
import cv2
import numpy as np

def generate_gradcam(image, face, model):
    tensor = transform(face).unsqueeze(0).to(DEVICE)

    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)

    heatmap = cam.generate(tensor)     # ✅ (H, W) float
    heatmap = cv2.resize(heatmap, (299, 299))

    # Create visualization
    heatmap_vis = np.uint8(255 * heatmap)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(face_bgr, 0.5, heatmap_vis, 0.5, 0)

    # ✅ IMPORTANT: return BOTH
    return overlay[:, :, ::-1], heatmap
