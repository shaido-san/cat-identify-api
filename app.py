import os
from flask import Flask, request, jsonify

from classifier import predict_category
from identifier import extract_features, match_candidates

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/identify', methods=['POST'])
def identify_cat():
    print("=== POST /identify 受信 ===")
    print("request.files:", request.files)
    print("request.form:", request.form)

    if 'image' not in request.files:
        print("▶️ 画像ファイルがありません")
        return jsonify({'error':'画像ファイルが必要です'}), 400
    
    image = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    category = predict_category(image)

    feature_vec = extract_features(image)
    match_candidates_list = match_candidates(feature_vec)


    response_data = {
        'match_candidates': match_candidates_list,
        'suggested_category': category
    }
    print("▶️ Flaskが返すデータ:", response_data)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)