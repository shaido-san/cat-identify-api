import numpy as np
import random

FAKE_DATABASE = {
    "abc123": np.random.rand(512),
    "def456": np.random.rand(512),
    "ghi789": np.random.rand(512),
}

def extract_features(image_file):
    """
    画像から特徴ベクトルを抽出（仮：ランダム）
    """
    # 実際はモデルを使って画像からベクトル抽出
    return np.random.rand(512)

def match_candidates(feature_vec, top_n=2):
    """
    ベクトルと既存猫たちの特徴を比較して類似度順に返す
    """
    results = []
    for cat_id, vec in FAKE_DATABASE.items():
        sim = cosine_similarity(feature_vec, vec)
        results.append((cat_id, sim))
    return [
        {"individual_id": cat_id, "confidence": round(sim, 2)}
        for cat_id, sim in results[:top_n]
    ]
 
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))