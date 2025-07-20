import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

LABELS_URL =  "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(LABELS_URL) as response:
    categorical = json.load(response)

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probs, 5)
    return [(categorical[str(idx.item())][1], float(probs[idx])) for idx in top5.indices]

if __name__ == "__main__":
    import sys
    results = predict(sys.argv[1])
    for label, score in results:
        print(f"{label}: {score*100:.1f}%")