import torch
import torchvision.transforms as transforms
from PIL import Image

def load_category_model():
    # ここで保存済みもですを読み込むコードを追加
    # 今はなし
    return None

def predict_category(image_file, model=None):
    """
    画像ファイル（FlaskのFileStorage）を受け取り、カテゴリを推定する
    """
    # ※本来は画像をPILに変換して前処理して推論するが
    # ここでは仮で '茶トラ' を返す
    return "茶トラ"