import os
import glob
from PIL import Image

def parse_annotation(file_path):
    with open(file_path, 'r') as f:
        parts = list(map(int, f.read().strip().split()))
        points = list(zip(parts[1::2], parts[2::2]))
        return points

def crop_face(image_path, points, save_path):
    img = Image.open(image_path)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin_x = int((max_x - min_x) * 0.3)
    margin_y = int((max_y - min_y) * 0.3)

    crop_box = (
        max(min_x - margin_x, 0),
        max(min_y - margin_y, 0),
        min(max_x + margin_x, img.width),
        min(max_y + margin_y, img.height)
    )

    cropped = img.crop(crop_box)
    cropped.save(save_path)
    print(f" Saved face to {save_path}")

def process_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cat_files = glob.glob(os.path.join(data_dir, "**/*.cat"), recursive=True)

    for cat_file in cat_files:
        image_path = cat_file.replace(".cat", "")
        if not os.path.exists(image_path):
            continue
        points = parse_annotation(cat_file)
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, image_name)
        crop_face(image_path, points, save_path)

if __name__ == "__main__":
    DATASET_DIR = "archive"
    OUTPUT_DIR = "cat_dataset/faces"
    process_dataset(DATASET_DIR, OUTPUT_DIR)