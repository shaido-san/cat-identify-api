import os, json, torch
from pathlib import Path
import torchvision.transforms as T
import torchvision import models
from PIL import Image
from torch.utils.data import DataLoader

MODEL_PATH = Path("models/cat_clasifier.pth")

def load_category_model():
    if not MODEL_PATH.exists():
        return None
    
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt("class_names")

    # 事前学習済み ResNet18 をベースに、fc をクラス数に差し替え
    base = models.resnet18(weights=models.ResNET18_Weights.DEFAULT)
    in_features = base.fc.in_features
    base.fc = torch.nn.Linear(in_features, len(class_names))
    base.load_state_dict(ckpt["model_state"])
    base.eval()
    return base, class_names

def predict_category(image_file, model=None):
    """
    画像ファイル（FlaskのFileStorage）を受け取り、カテゴリを推定する
    """
    # ※本来は画像をPILに変換して前処理して推論するが
    # ここでは仮で '茶トラ' を返す
    return "茶トラ"