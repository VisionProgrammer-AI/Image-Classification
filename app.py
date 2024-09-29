import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from werkzeug.utils import secure_filename
import os
from PIL import Image
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(28*28, 128),
                      nn.ReLU(),
                      nn.Linear(128,10)
                      )

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((28,28)),
                                transforms.ToTensor()]
                               )

@app.route('/')
def index():
    return render_template(('index.html'))
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('index.html', filename = filename)
    return redirect(url_for('index'))

@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(filepath)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return  render_template('index.html', filename=filename, prediction= prediction)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+ filename), code=301)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)