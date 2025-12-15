from flask import Flask, render_template, request, jsonify
import os
import io
import base64
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.FPN(
    encoder_name='vgg16',
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("model/FPN_vgg16_best_weights.pth", map_location=device))
model.eval()


def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def resize_to_256(image: Image.Image) -> Image.Image:
    target_size = (256, 256)
    return image.resize(target_size, Image.LANCZOS)


def dummy_process_image(image: Image.Image) -> tuple[Image.Image, str]:
    message = "Обработка завершена"
    return image, message
    
    
def predict_mask(image: Image.Image) -> Image.Image:
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0]

    mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    mask_rgb[..., 0] = pred_mask * 255
    mask_img = Image.fromarray(mask_rgb)

    resized_img = image.resize((256, 256))
    overlay = Image.blend(resized_img, mask_img, alpha=0.4)
    return overlay

def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    filename = file.filename

    if not allowed_file(filename):
        return jsonify({'error': 'Разрешены только PNG/JPG/JPEG/TIF/TIFF файлы'}), 400

    try:
        name_lower = filename.lower()

        if name_lower.endswith((".tif", ".tiff")):
            # Для TIF-файлов используем полноценное цветное изображение
            image = Image.open(file.stream)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
        else:
            image = Image.open(file.stream).convert("RGB")

        resized = resize_to_256(image)

        processed_image = predict_mask(resized)

        safe_name = secure_filename(file.filename)
        base_name, _ = os.path.splitext(safe_name)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_input.png")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_processed.png")

        resized.save(input_path, format="PNG")
        processed_image.save(output_path, format="PNG")

        processed_b64 = image_to_base64_png(processed_image)

        return jsonify({
            'message': "Обработка завершена",
            'processed_image': processed_b64
        }), 200
    except Exception as exc:
        return jsonify({'error': f'Ошибка обработки изображения: {exc}'}), 500


@app.route('/preview', methods=['POST'])
def preview():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    filename = file.filename

    if not allowed_file(filename):
        return jsonify({'error': 'Разрешены только PNG/JPG/JPEG/TIF/TIFF файлы'}), 400

    try:
        name_lower = filename.lower()

        if name_lower.endswith((".tif", ".tiff")):
            # Для TIF-файлов используем полноценное цветное изображение
            image = Image.open(file.stream)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
        else:
            image = Image.open(file.stream).convert("RGB")

        resized = resize_to_256(image)
        preview_b64 = image_to_base64_png(resized)

        return jsonify({'preview_image': preview_b64}), 200
    except Exception as exc:
        return jsonify({'error': f'Ошибка предпросмотра изображения: {exc}'}), 500


if __name__ == '__main__':
    print("Запуск приложения...")
    print("Откройте браузер и перейдите по адресу: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)




