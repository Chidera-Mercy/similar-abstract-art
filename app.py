from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Directory to save uploaded files

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_image(image, max_dim=512):
    img = Image.open(image)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (max_dim, max_dim), preserve_aspect_ratio=True)
    img = img[tf.newaxis, :]
    return img

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def fine_tune_model(model, content_image, style_image):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    content_image = request.files['content_image']
    style_image = request.files['style_image']
    
    # Save the content image
    content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded-content-image.png')
    content_image.save(content_image_path)

    # Load images
    content_image = load_image(content_image)
    style_image = load_image(style_image)

    # Apply style transfer
    stylized_image = fine_tune_model(hub_model, content_image, style_image)
    
    # Save the stylized image
    stylized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated-stylized-image.png')
    stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)[0]
    stylized_image = Image.fromarray(stylized_image.numpy())
    stylized_image.save(stylized_image_path, format='PNG')

    # Provide paths to images
    return render_template('index.html', 
                           stylized_image_url='/static/uploads/generated-stylized-image.png',
                           content_image_url='/static/uploads/uploaded-content-image.png')

if __name__ == '__main__':
    app.run(debug=True)
