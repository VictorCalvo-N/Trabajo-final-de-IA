from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import time
import hashlib
from PIL import Image
import io
import base64
import torch

from transformer import ImageProcessor

# Configuración
app = Flask(__name__, static_folder='.')
CORS(app)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuración de carpetas y límites
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = 'cache'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB

# Crear carpetas necesarias
for folder in [UPLOAD_FOLDER, CACHE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_hash(file_content):
    """Genera un hash único para el contenido del archivo."""
    return hashlib.md5(file_content).hexdigest()

# Cache para el modelo
model_instance = None
def get_model():
    global model_instance
    if model_instance is None:
        model_instance = ImageProcessor()
    return model_instance

@app.route('/')
def serve_html():
    return send_file('main.html')

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        # Validación inicial
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

        # Leer el contenido del archivo y generar hash
        file_content = file.read()
        file_hash = get_file_hash(file_content)
        cache_path = os.path.join(CACHE_FOLDER, f"{file_hash}.png")

        # Verificar caché
        if os.path.exists(cache_path):
            logging.info(f"Imagen encontrada en caché: {file_hash}")
            with open(cache_path, 'rb') as f:
                img_data = f.read()
                img_str = base64.b64encode(img_data).decode()
                return jsonify({
                    'success': True,
                    'processed_image': f'data:image/png;base64,{img_str}',
                    'from_cache': True
                })

        # Procesar nueva imagen
        logging.info(f"Iniciando procesamiento de nueva imagen: {file_hash}")
        start_time = time.time()

        # Validar tamaño de imagen
        img = Image.open(io.BytesIO(file_content))
        if img.size[0] * img.size[1] > 2048 * 2048:
            return jsonify({'error': 'Image too large. Maximum size: 2048x2048'}), 400

        # Guardar imagen temporal
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(temp_path)

        try:
            # Procesar imagen
            processor = get_model()
            processed_image = processor.process_image(temp_path)
            
            # Guardar en caché
            processed_image.save(cache_path, format="PNG")
            
            # Convertir a base64
            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            logging.info(f"Procesamiento completado en {processing_time:.2f} segundos")

            return jsonify({
                'success': True,
                'processed_image': f'data:image/png;base64,{img_str}',
                'processing_time': f"{processing_time:.2f}",
                'from_cache': False
            })

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return jsonify({'error': 'Out of GPU memory. Try with a smaller image'}), 500
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Error processing image'}), 500
        finally:
            # Limpieza de archivos temporales
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Endpoint para limpiar la caché de imágenes procesadas."""
    try:
        files_removed = 0
        for filename in os.listdir(CACHE_FOLDER):
            file_path = os.path.join(CACHE_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
        
        return jsonify({
            'success': True,
            'message': f'Cache cleared. {files_removed} files removed.'
        })
    except Exception as e:
        return jsonify({
            'error': f'Error clearing cache: {str(e)}'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File too large. Maximum size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB'}), 413

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    app.run(debug=True, port=5000)