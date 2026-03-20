import torch

class ModelLoader:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        # 🔥 PUT IT HERE
        model = torch.load(
            "model/model.pth",
            map_location="cpu",
            weights_only=False
        )

        model.eval()
        torch.set_num_threads(1)

        return model

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)

model_loader = ModelLoader()