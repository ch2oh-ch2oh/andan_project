from flask import Flask, render_template, request, jsonify
import os
import io
import base64
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Создаем папку для загрузок, если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Разрешенные расширения
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def resize_to_256(image: Image.Image) -> Image.Image:
    """
    Сжать изображение до ровно 256x256 без обрезки.

    Масштабирует картинку по обеим осям до 256x256, возможно с изменением пропорций.
    """
    target_size = (256, 256)
    return image.resize(target_size, Image.LANCZOS)


def dummy_process_image(image: Image.Image) -> tuple[Image.Image, str]:
    """
    Заглушка «обработки» изображения.

    Здесь можно будет подключить вашу реальную модель / логику.
    Сейчас:
    - ждём 2 секунды, чтобы показать статус «обрабатывается»;
    - переводим картинку в оттенки серого.
    """
    time.sleep(2)
    processed = image.convert("L").convert("RGB")  # пример: ч/б
    message = "Обработка завершена: изображение переведено в оттенки серого."
    return processed, message


def image_to_base64_png(image: Image.Image) -> str:
    """Преобразовать PIL.Image в base64-строку PNG."""
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

    if not allowed_file(file.filename):
        return jsonify({'error': 'Разрешены только PNG/JPG/JPEG файлы'}), 400

    try:
        # Читаем изображение в PIL
        image = Image.open(file.stream).convert("RGB")

        # Приводим к размеру 256x256
        resized = resize_to_256(image)

        # Здесь вызываем «метод» обработки (пока заглушка)
        processed_image, result_text = dummy_process_image(resized)

        # Сохраняем (опционально) на диск исходник и результат
        safe_name = secure_filename(file.filename)
        base_name, _ = os.path.splitext(safe_name)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_input.png")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_processed.png")

        resized.save(input_path, format="PNG")
        processed_image.save(output_path, format="PNG")

        # Возвращаем обработанное изображение как base64 и строку
        processed_b64 = image_to_base64_png(processed_image)

        return jsonify({
            'message': result_text,
            'processed_image': processed_b64
        }), 200
    except Exception as exc:
        return jsonify({'error': f'Ошибка обработки изображения: {exc}'}), 500


if __name__ == '__main__':
    print("Запуск приложения...")
    print("Откройте браузер и перейдите по адресу: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)




