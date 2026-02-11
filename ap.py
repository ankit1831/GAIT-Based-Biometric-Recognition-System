from collections import Counter

def predict_subjects_in_folder_majority(folder_path, model, device):
    folder_path = Path(folder_path)
    image_paths = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))

    if not image_paths:
        print(f"No images found in folder {folder_path}")
        return None

    predicted_classes = []

    for img_path in image_paths:
        # Preprocess using your exact process
        preprocessed_img = preprocess_image(str(img_path))
        if preprocessed_img is None:
            print(f"Skipping image (preprocess failed): {img_path.name}")
            continue

        # Convert to RGB tensor
        rgb_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        tensor = transforms.ToTensor()(rgb_img)
        normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        tensor = normalize(tensor)
        tensor = tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        predicted_classes.append(pred_class)

        print(f"Image {img_path.name} predicted as Subject_{pred_class:03d}")

    if not predicted_classes:
        print("No valid predictions.")
        return None

    # Count most frequent prediction
    counter = Counter(predicted_classes)
    most_common_subject, count = counter.most_common(1)[0]

    print(f"\nMost frequent predicted subject across folder: Subject_{most_common_subject:03d} (predicted {count} times)")
    return most_common_subject



import cv2
from pathlib import Path
import numpy as np


def preprocess_image(image_path, output_size=(72, 72)):

    try:
        # 1. Load image as grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # 2. Find bounding box of silhouette
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"Warning: No silhouette found, returning resized original")
            return cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
        
        # 3. Crop to bounding box
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_img = img[y:y+h, x:x+w]
        
        # 4. Resize to 72x72
        resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)
        
        return resized_img
    
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def load_model(model_path, num_classes=123):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Running on: {device}")
    
    return model, device


def predict_subject(image_path, model, device):
    # Step 1: Preprocess image
    print(f"\n1. Preprocessing image: {image_path}")
    preprocessed = preprocess_image(image_path)
    
    if preprocessed is None:
        return {'error': 'Failed to preprocess image'}
    
    # Step 2: Convert to tensor
    print(f"2. Converting to tensor...")
    # Convert grayscale to RGB (3 channels)
    rgb_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    tensor = transforms.ToTensor()(rgb_img)  # (3, 72, 72)
    
    # Normalize with same values as training
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = normalize(tensor)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)  # (1, 3, 72, 72)
    tensor = tensor.to(device)
    
    # Step 3: Inference
    print(f"3. Running inference...")
    with torch.no_grad():
        outputs = model(tensor)
    
    # Step 4: Get predictions
    print(f"4. Processing results...")
    softmax_scores = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence, predicted_class = torch.max(softmax_scores, dim=0)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    all_confidences = softmax_scores.cpu().numpy()
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_confidences': all_confidences
    }

def display_results(result, image_path=None):
    print("GAIT RECOGNITION PREDICTION")
    
    if image_path:
        print(f"Image: {Path(image_path).name}")
    
    if 'error' in result:
        print(f" Error: {result['error']}")
    else:
        print(f"\n✓ Predicted Subject: Subject_{result['predicted_class']:03d}")
        print(f"✓ Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        # Top 3 predictions
        top_3_idx = np.argsort(result['all_confidences'])[-3:][::-1]
        print(f"\nTop 3 Predictions:")
        for rank, idx in enumerate(top_3_idx, 1):
            print(f"  {rank}. Subject_{idx:03d}: {result['all_confidences'][idx]:.4f}")
    


import cv2
from pathlib import Path

def extract_frames_from_video(video_path, output_folder, frame_rate=10):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    frame_paths = []
    count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save one frame every frame_interval
        if count % frame_interval == 0:
            frame_path = output_folder / f"frame_{saved_frame_count:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_frame_count += 1
        
        count += 1

    cap.release()

    print(f"Extracted {saved_frame_count} frames from video.")
    return frame_paths

def convert_frames_to_silhouettes(frame_paths, silhouette_folder):
    silhouette_folder = Path(silhouette_folder)
    silhouette_folder.mkdir(parents=True, exist_ok=True)

    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    silhouette_paths = []

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        fg_mask = back_sub.apply(frame)
        fg_mask = back_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  # 200 is more strict than 127


        # Optional: Morphological operations to clean mask
          
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        

        # Save the silhouette image as PNG (grayscale)
        silhouette_path = silhouette_folder / f"silhouette_{i:04d}.png"
        cv2.imwrite(str(silhouette_path), fg_mask)
        silhouette_paths.append(silhouette_path)

    print(f"Converted {len(silhouette_paths)} frames to silhouettes.")
    return silhouette_paths






if __name__ == "__main__":
    video_path = r"D:\vit study\Machine Learning\Gait\WhatsApp Video 2025-12-11 at 15.38.16_fd73e292.mp4"
    frames_folder = "temp_frames"
    silhouettes_folder = "temp_silhouettes"

    frame_paths = extract_frames_from_video(video_path, frames_folder, frame_rate=10)

    silhouette_paths = convert_frames_to_silhouettes(frame_paths, silhouettes_folder)

    model, device = load_model("gait_simple_cnn_model_123_subjects.pth", num_classes=123)
    results = predict_subjects_in_folder_majority(silhouettes_folder, model, device)

    silhouette_folder = "temp_silhouettes" 



