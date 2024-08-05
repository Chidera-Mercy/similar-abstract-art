# Importing necessary modules from Flask and others
from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

# Initializing the Flask application
app = Flask(__name__)

# Directory to save uploaded files
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Directory to save uploaded files

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_image(image, max_dim=512):
    """
    Load an image, convert to RGB, resize maintaining aspect ratio, and prepare for TensorFlow.
    :param image: Image file
    :param max_dim: Maximum dimension to resize the image
    :return: Processed image tensor
    """
    img = Image.open(image)  # Open the image
    img = img.convert('RGB')  # Convert image to RGB
    img = np.array(img)  # Convert image to a numpy array
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert image dtype to float32
    img = tf.image.resize(img, (max_dim, max_dim), preserve_aspect_ratio=True)  # Resize image
    img = img[tf.newaxis, :]  # Add a new axis for batch dimension
    return img  # Return the processed image

# Load the model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def generate_similar_image(model, content_image, style_image):
    """
    Apply style transfer using the loaded model.
    :param model: Loaded TensorFlow Hub model
    :param content_image: Content image tensor
    :param style_image: Style image tensor
    :return: Stylized image tensor
    """
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]  # Apply style transfer
    return stylized_image  # Return the stylized image

@app.route('/')
def index():
    """
    Render the index page.
    """
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    """
    Handle the image stylization request.
    """
    content_image = request.files['content_image'] # Get the content image from the request
    style_image = request.files['style_image'] # Get the style image from the request
    
    # Save the content image
    content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded-content-image.png')
    content_image.save(content_image_path) # Save the uploaded content image

    # Save the style image
    style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded-style-image.png')
    style_image.save(style_image_path) # Save the uploaded style image

    # Load images
    content_image = load_image(content_image) # Load and process the content image
    style_image = load_image(style_image) # Load and process the style image

    # Apply style transfer
    stylized_image = generate_similar_image(hub_model, content_image, style_image) # Generate stylized image
    
    # Save the stylized image
    stylized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated-stylized-image.png')
    stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)[0] # Convert stylized image to uint8
    stylized_image = Image.fromarray(stylized_image.numpy()) # Convert stylized image to uint8
    stylized_image.save(stylized_image_path, format='PNG') # Save the stylized image as PNG

    # Provide paths to images in the response
    return render_template('index.html', 
                           stylized_image_url='/static/uploads/generated-stylized-image.png',
                           content_image_url='/static/uploads/uploaded-content-image.png',
                           style_image_url='/static/uploads/uploaded-style-image.png')

if __name__ == '__main__':
    app.run(debug=True) # Run the app in debug mode
