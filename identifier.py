import torch
import io
import json
import os
from scipy.spatial.distance import cosine
from torchvision import models, transforms
from PIL import Image
import numpy as np

def extract_features(image_file):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_file.stream).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        features = output.squeeze().numpy()
    
    return features.tolist()

CAT_DATABASE = "cat_database.json"

def register_cat(image_file, individual_id):
    features = extract_features(image_file)

    base_dir = os.path.join("db", individual_id)
    image_dir = os.path.join(base_dir, "images")
    feature_dir = os.path.join(base_dir, "features")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    next_id = len(existing_images) + 1

    image_filename = f"{next_id}.jpg"
    feature_filename = f"{next_id}.json"

    image_path = os.path.join(image_dir, image_filename)
    feature_path = os.path.join(feature_dir, feature_filename)

    image = Image.open(image_file.stream).convert("RGB")
    image.save(image_path)

    with open(feature_path, "w") as f:
        json.dump(features, f)
    
    return {
        "message": f"{individual_id}ちゃんを登録しました!",
        "image_path": image_path,
        "feature_path": feature_path
    }

def identify_cat(image_file):
    features = extract_features(image_file)

    if not os.path.exists(CAT_DATABASE):
        return {"message": "データベースがまだ空っぽです"}
    
    with open(CAT_DATABASE, "r") as f:
        database = json.load(f)
    
    min_distance = float("inf")
    best_match = None

    for cat_name, feature_list in database.items():
        for saved_feature in feature_list:
            dist = cosine(features, saved_feature)
            if dist < min_distance:
                min_distance = dist
                best_match = cat_name
    
    if best_match is None:
        return {"message": "一致する猫ちゃんが見つかりませんでした"}
    
    return {
        "cat": best_match,
        "similarity": f"{(1 - min_distance) * 100:.2f}%"
    }

def save_cat_feature(name, feature_vector):
    os.makedirs("db", exist_ok=True)
    db_path = os.path.join("db", f"{name}.json")

    with open(db_path, "w") as f:
        json.dump(feature_vector.tolist(), f)
    
    print(f"{name}の特徴を保存しました: {db_path}")

def match_candidates(input_feature, top_n=3):
    db_dir = "db"
    results = []

    for individual in os.listdir(db_dir):
        feature_dir = os.path.join(db_dir, individual, "features")
        image_dir = os.path.join(db_dir, individual, "images")

        if not os.path.isdir(feature_dir) or not os.path.isdir(image_dir):
            continue

        for filename in os.listdir(feature_dir):
           if not filename.endswith(".json"):
            continue
        
        feature_path = os.path.join(feature_dir, filename)
        with open(feature_path, "r") as f:
            saved_vector = json.load(f)
        
        distance = cosine(input_feature, saved_vector)
        similarity = 1 - distance

        image_filename = filename.replace(".json", ".jpg")
        image_path = os.path.join(image_dir, image_filename)

        results.append({
            "individual_id": individual,
            "confidence": f"{similarity * 100:.2f}%",
            "image_path": image_path
        })

    results.sort(key=lambda x: float(x["confidence"].rstrip("%")), reverse=True)
    return results[:top_n]