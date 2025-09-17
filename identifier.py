import torch
import json
import os
import numpy as np
from scipy.spatial.distance import cosine
from torchvision import models, transforms
from PIL import Image
from urllib.parse import quote


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, os.path.join(BASE_DIR, "db"))
os.makedirs(DB_DIR, exist_ok=True)

try:
    RESNET = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    RESNET = models.resnet18(pretrained=True)
except Exception:
    RESNET = torch.nn.Sequential(*list(RESNET.children())[:-1])
    RESNET.eval()
    PREPROCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def open_pil(image_file):
    if hasattr(image_file, "stream"):
        return Image.open(image_file.stream).convert("RGB")
    if isinstance(image_file, (str, os.PathLike)):
        return Image.open(image_file).convert("RGB")
    return Image.open(image_file).convert("RGB")

def extract_features(image_file):
    image = open_pil(image)
    input_tensor = PREPROCESS(image).unsqueeze(0)
    with torch.no_grad:
        output = RESNET(input_tensor)
        features = output.squeeze().cpu().numpy()
    return features.tolist()

def individual_dif(individual_id: str) -> str:
    image_dir = os.path.join(DB_DIR, individual_id, "images")
    feature_dir = os.path.join(DB_DIR, individual_id, "features")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    return os.path.join(DB_DIR, individual_id)

def centroid_path(individual_id: str) -> str:
    return os.path.join(DB_DIR, individual_id, "centroid.json")

def load_centroid(individual_id: str):
    path = centroid_path(individual_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        vec = np.array(data.get("vector", []), dtype=np.float32)
        n = int(data.get("n", 0))
        if vec.size == 0 or n<= 0:
            return vec, n
    except Exception:
        return None

def save_centroid(individual_id: str, vector: np.ndarray, n: int):
    payload = {"vector": vector.tolist(), "n": int(n)}
    with open(centroid_path(individual_id), "w") as f:
        json.dump(payload, f, ensure_ascii=False)

def update_centroid_incremental(individual_id: str, new_vec: np.ndarray):
    loaded = load_centroid(individual_id)
    if loaded is None:
        save_centroid(individual_id, new_vec, 1)
        return
    vec, n = loaded
    new_n = n + 1
    new_centroid = (vec * n + new_vec) / new_n
    save_centroid(individual_id, new_centroid, new_n)

def register_cat(image_file, individual_id):
    features = np.array(extract_features(image_file), dtype=np.float32)

    base_dir = individual_id(individual_id)
    image_dir = os.path.join(base_dir, "images")
    feature_dir = os.path.join(base_dir, "features")

    jpg_files = []
    for f in os.listdir(image_dir):
        if f.endswith(".jpg"):
            jpg_files.append(f)
    
    image_numbers = []
    for filename in jpg_files:
        name_part = filename.split(".")[0]
        if name_part.isdigit():
            image_numbers.append(int(name_part))
    
    if len(image_numbers) == 0:
        next_id = 1
    else:
        next_id = max(image_numbers) + 1
    
    image_path = os.path.join(image_dir, f"{next_id}.jpg")
    feature_path = os.path.join(feature_dir, f"{next_id}.json")

    if hasattr(image_file, "stream"):
        try:
            image_file.stream.seek(0)
        except Exception:
            pass
    
    img = open_pil(image_file)
    img.save(image_path)

    with open(feature_path, "w") as f:
        json.dump(features.tolist(), f)
    
    update_centroid_incremental(individual_id, features)

    return {
        "message": f"{individual_id}ちゃんを登録しました！",
        "image_path": image_path,
        "feature_path": feature_path,
    }

def match_candidates(input_feature, top_n=3, threshold=None):
    if isinstance(input_feature, list):
        q = np.array(input_feature, dtype=np.float32)
    else:
        q = np.array(input_feature, dtype=np.float32)

    results = []

    for individual in os.listdir(DB_DIR):
        indiv_dir = os.path.join(DB_DIR, individual)
        feature_dir = os.path.join(indiv_dir, "features")
        image_dir = os.path.join(indiv_dir, "images")
        if not (os.path.isdir(feature_dir) and os.path.isdir(image_dir)):
            continue

        centroid = load_centroid(individual)
        if centroid is not None:
            cvec, _n = centroid
            score = 1.0 - float(cosine(q, cvec))
        else:
            best = -1.0
            for filename in os.listdir(feature_dir):
                if not filename.endswith(".json"):
                    continue
                with open(os.path.join(feature_dir, filename), "r") as f:
                    saved_vector = np.array(json.load(f), dtype=np.float32)
                s = 1.0 - float(cosine(q, saved_vector))
                if s > best:
                    best = s
            if best < 0.0:
                score = 0.0
            else:
                score = best

        rep_img = recent_image_path(image_dir)
        if rep_img is None:
            continue

        rel = os.path.relpath(rep_img, DB_DIR).replace("\\", "/")
        image_url = "/media/" + quote(rel)

        if (threshold is None) or (score >= float(threshold)):
            results.append({
                "individual_id": individual,
                "confidence": float(score),
                "image_path": rep_img,
                "image_url": image_url,
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_n]


