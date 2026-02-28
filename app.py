import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# ==========================================
# 1. DEFINE THE AI MODEL (Same as training)
# ==========================================
class LumaNet(nn.Module):
    def __init__(self):
        super(LumaNet, self).__init__()
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True) 
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image = x + r4 * (torch.pow(x, 2) - x)
        return enhanced_image

# ==========================================
# 2. SETUP FLASK & LOAD MODEL
# ==========================================
app = Flask(__name__)

# Folders for images
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the Brain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LumaNet().to(device)

# Load weights safely (map_location ensures it runs on CPU even if trained on GPU)
if os.path.exists("weights.pth"):
    model.load_state_dict(torch.load("weights.pth", map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully!")
else:
    print("❌ Error: weights.pth not found! Put it in the same folder.")

# ==========================================
# 3. WEB ROUTES
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # A. Check if file is uploaded
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # B. Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # C. Run AI Enhancement
        output_filename = "enhanced_" + filename
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        process_image(input_path, output_path)

        # D. Show Result
        return render_template('index.html', 
                               original_img=input_path, 
                               enhanced_img=output_path)

    return render_template('index.html', original_img=None, enhanced_img=None)

def process_image(img_path, save_path):
    # 1. READ IMAGE
    img = cv2.imread(img_path)
    if img is None: return

    # 2. SMART RESIZE (Keep original quality)
    original_h, original_w = img.shape[:2]
    max_dim = 1024 
    scale = min(max_dim / original_w, max_dim / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    new_w = new_w - (new_w % 4)
    new_h = new_h - (new_h % 4)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 3. AI INFERENCE
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # Convert back to Image
    result = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1) * 255.0
    result = result.astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # =======================================================
    # 🔴 FIX 1: GAMMA CORRECTION (Kills the Fog)
    # =======================================================
    # Values > 1.0 darken the shadows. We use 1.2 to 1.5 to remove the grey haze.
    gamma = 1.3 
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    result_bgr = cv2.LUT(result_bgr, lookUpTable)

    # =======================================================
    # 🔴 FIX 2: UNSHARP MASKING (Fixes the Blur)
    # =======================================================
    # Instead of Denoising (which blurs), we SHARPEN using Gaussian subtraction
    gaussian = cv2.GaussianBlur(result_bgr, (0, 0), 3.0)
    result_sharp = cv2.addWeighted(result_bgr, 1.5, gaussian, -0.5, 0)
    
    # =======================================================
    # 🔴 FIX 3: COLOR BALANCE (Fixes Green Tint)
    # =======================================================
    # Smart Contrast (CLAHE)
    lab = cv2.cvtColor(result_sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Boost Saturation slightly
    hsv = cv2.cvtColor(final_result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15) 
    final_result = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # 4. SAVE
    cv2.imwrite(save_path, final_result)
    
if __name__ == '__main__':
    app.run(debug=True)


# .\.venv\Scripts\Activate.ps1    