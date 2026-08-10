# model_handler.py
import torch
import torch.nn as nn
import os
import sys
from ts.torch_handler.base_handler import BaseHandler

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "models"))

from linear import SimpleLinearModel


class SimpleModelHandler(BaseHandler):
    """
    Custom handler for SimpleLinearModel
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Initialize model. This will be called during model loading time"""
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        serialized_file = "model.pth"
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Initialize model
        self.model = SimpleLinearModel()
        state_dict = torch.load(model_pt_path, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True
        return self

    def preprocess(self, data):
        # Get the value directly without decoding
        value = float(data[0].get("body"))
        tensor = torch.tensor([value], dtype=torch.float32).view(1, 1)
        return tensor.to(self.device)

    def inference(self, data):
        """Run inference on the preprocessed data"""
        with torch.no_grad():
            results = self.model(data)
        return results

    def postprocess(self, inference_output):
        """Return inference result"""
        return inference_output.tolist()


_service = SimpleModelHandler()
