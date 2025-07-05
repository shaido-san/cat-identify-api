import os
import hashlib
from flask import Flask, request, jsonify

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

    match_candidates = [
        {'individual_id': 'abc123', 'confidence': 0.87},
        {'individual_id': 'def456', 'confidence': 0.75}
    ]

    response_data = {
        'match_candidates': match_candidates,
        'suggested_category': '茶トラ'
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)