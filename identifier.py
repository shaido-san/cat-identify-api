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

def register_cat(image_file, cat_name):
    features = extract_features(image_file)

    if not os.path.exists(CAT_DATABASE):
        with open(CAT_DATABASE, "w") as f:
            json.dump({}, f)
    
    with open(CAT_DATABASE, "r") as f:
        database = json.load(f)
    
    if cat_name in database:
        database[cat_name].append(features)
    else:
        database[cat_name] = [features]
    
    with open(CAT_DATABASE, "w") as f:
        json.dump(database, f)
    
    return {
        "mesage": f"{cat_name}ちゃんを登録しました!",
        "count": len(database[cat_name])
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
    
    print(f"{name}の特商を保存しました: {db_path}")

def match_candidates(input_feature, top_n=3):
    db_dir = "db"
    results = []

    for filename in os.listdir(db_dir):
        if not filename.endswith(".json"):
            continue

        name = filename[:-5]
        with open(os.path.join(db_dir, filename), "r") as f:
            saved_vector = json.load(f)
        
        distance = cosine(input_feature, saved_vector)
        similarity = 1 - distance

        results.append({
            "individual_id": name,
            "confidence": f"{similarity * 100:.2f}%",
            "image_path": f"db/{name}/main.jpg"
        })

    results.sort(key=lambda x: float(x["confidence"].rstrip("%")), reverse=True)
    return results[:top_n]