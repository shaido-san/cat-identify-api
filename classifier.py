import torch
from pathlib import Path
import torchvision.transforms as T
import torchvision import models
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" /"cat_classifier.pth"

def load_category_model():
    if not MODEL_PATH.exists():
        return None
    
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    class_names = ckpt["class_names"]

    # 事前学習済み ResNet18 をベースに、fc をクラス数に差し替え
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = base.fc.in_features
    base.fc = torch.nn.Linear(in_features, len(class_names))
    base.load_state_dict(ckpt["model_state"])
    base.eval()
    return base, class_names

def predict_category(image_file, model=None):
   
   img = Image.open(image_file.stream).convert("RGB")
   model_info = model or load_category_model()
   if model_info is None:
       return "その他"
   
   model, class_names = model_info
   
   transform = T.Compose([
       T.Resize(256),
       T.CenterCrop(224),
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
   ])
   tensor = transform(img).unsqueeze(0)

   with torch.no_grad():
       outputs = model(tensor)
       pred_idx = outputs.argmax(dim=1).item()
    
   return class_names[pred_idx]