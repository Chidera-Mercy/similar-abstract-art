Project Name: Abstract Art Similar Images

Project Overview:
This project utilizes style transfer techniques to generate abstract art images that are similar to a given content image but in a different style.
The project enables users to create unique abstract art by transferring styles from one image to another, allowing for artistic creativity and exploration.

How It Works:
- Input Images:
        Content Image: The abstract art image you want to stylize.
        Style Image: The image whose style you want to apply to the content image.
- Process:
The project generates an image that combines the content of the content image with the style of the style image while preserving the key features of the content image. This is achieved using the Arbitrary Image Stylization v1-256 model from TensorFlow Hub. This model allows you to apply the style of one image to the content of another, resulting in a stylized output. It is part of Google's Magenta project, which explores the use of machine learning in art and music.

Deployment:
The application is deployed using Flask, providing a beautiful and intuitive web interface for users.

Hosting the Application Locally:

1. Ensure all dependencies are installed. You need Flask, TensorFlow, TensorFlow Hub, NumPy, PIL, and OS. Install them via your system's terminal or an IDE terminal.

2. Navigate to the folder containing the static folder, templates folder, and app.py. The folder structure should look like this:
project-folder/
├── static/...
├── templates/...
└── app.py

3. Run the following command in your terminal: python app.py
