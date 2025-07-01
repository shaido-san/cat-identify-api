import os
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/identify', methods=['POST'])
def identify_cat():
    if 'image' not in request.files:
        return jsonify({'error': '画像ファイルが必要です'}), 400
    
    image = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    return jsonify({
        'individual_id': file_hash,
        'category': 'その他'
    })

if __name__ == '__main__':
    app.run(debug=True)