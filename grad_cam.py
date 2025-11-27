import cv2
import torch
import numpy as np
import torch.nn.functional as F

# -----------------------------
# GRAD-CAM for EfficientNet-B0
# -----------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        input_tensor: torch tensor of shape [1, 3, 299, 299] (already transformed)
        """
        # Forward pass
        output = self.model(input_tensor)

        # The model outputs a single logit → GradCAM usually takes class score
        score = output.squeeze()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # CAM computation
        gradients = self.gradients  # [bs, C, H, W]
        activations = self.activations  # [bs, C, H, W]

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP

        cam = (weights * activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.cpu().numpy()
        return cam

