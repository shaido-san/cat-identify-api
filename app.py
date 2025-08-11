import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from classifier import predict_category
from identifier import extract_features, match_candidates, register_cat

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
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
    filename = secure_filename(image.filename)
    if not filename:
        return jsonify({'error': 'ファイルが不正です'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
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

@app.route('/register', methods=['POST'])
def register_cat_route():
    if 'image' not in request.files or 'individual_id' not in request.form:
        return jsonify({'error':'画像とindividual_idが必要です'}), 400
    
    image = request.files['image']
    individual_id = request.form['individual_id']

    result = register_cat(image, individual_id)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
    