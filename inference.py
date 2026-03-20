import torch
from app.model import model_loader

CLASS_MAP = {
    0: "person_1",
    1: "person_2",
    # ADD ALL YOUR 123 CLASSES
}

def run_inference(sequence):
    x = torch.tensor(sequence, dtype=torch.float32)

    # IMPORTANT: adjust if your model needs channel dimension
    x = x.unsqueeze(0)  # (1, frames, H, W)

    outputs = model_loader.predict(x)

    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return CLASS_MAP[pred], confidence