import os
import torch
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define ResNet9 model
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))
        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Flask app initialization
app = Flask(__name__, static_folder='static', template_folder='templates')

# Blood group classes
class_names = ['A', 'B', 'AB', 'O']

# Model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet9(3, 8)  # Original model with 8 classes
model.load_state_dict(torch.load('FingurePrintTOBloodGroup.pth', map_location=device))

# Replace classifier for 4 classes
model.classifier[3] = nn.Linear(512, len(class_names))
torch.nn.init.xavier_uniform_(model.classifier[3].weight)
model.classifier[3].bias.data.fill_(0)

model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        
    return class_names[predicted.item()]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about-the-blood')
def about_the_blood():
    return render_template('AbouttheBlood.html')

@app.route('/blood-group-check')
def blood_group_check():
    return render_template('bloodGroupCheck.html')

@app.route('/our-method')
def our_method():
    return render_template('Ourmethod.html')

@app.route('/prediction-page')
def prediction_page():
    return render_template('prediction-page.html')

@app.route('/about-the-blood')
def about_blood():
    return render_template('AbouttheBlood.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        prediction = predict(file_path)

        os.remove(file_path)  # Clean up uploaded file

        return jsonify({'blood_group': prediction})

    except Exception as e:
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

# Run the Flask app
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
