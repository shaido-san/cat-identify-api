import torch
import json
import os
from scipy.spatial.distance import cosine
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ResNet18で画像特徴ベクトルを抽出（PIL + torch）
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

# 個体ごとの画像と特徴ベクトルをDBに保存
def register_cat(image_file, individual_id):
    features = extract_features(image_file)

    base_dir = os.path.join("db", individual_id)
    image_dir = os.path.join(base_dir, "images")
    feature_dir = os.path.join(base_dir, "features")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    image_files = os.listdir(image_dir)
    jpg_files = [f for f  in image_files if f.endswith(".jpg")]

    image_numbers = []
    for filename in jpg_files:
        name_part = filename.split('.')[0]
        if name_part.isdigit():
            image_numbers.append(int(name_part))
    
    next_id = max(image_numbers, default=0) + 1

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


# 入力ベクトルに最も近い候補をDBから探して返す
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
               "confidence": similarity,
               "image_path": image_path
           })

    results.sort(key=lambda x: float(x["confidence"].rstrip("%")), reverse=True)
    return results[:top_n]