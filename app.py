from flask import Flask, render_template, request, url_for
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import os
import torch.nn as nn
from torchvision.models import efficientnet_b0

# ---------------- Flask setup ----------------
app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Model ----------------
# Load EfficientNet-B0 architecture
num_classes = 27  # replace with your number of classes
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load your trained weights
# Relative path (simplest)
model.load_state_dict(torch.load("models/efficientnet_b0_best.pth", map_location=device))
model = model.to(device)
model.eval()  # set to evaluation mode

# ---------------- Classes ----------------
classes = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism", 
    "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Contemporary_Realism", "Cubism", "Early_Renaissance",
    "Expressionism", "Fauvism", "High_Renaissance", "Impressionism",
    "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
    "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art",
    "Post_Impressionism", "Realism", "Rococo", "Romanticism",
    "Symbolism", "Synthetic_Cubism", "Ukiyo_e"
]

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url")

        try:
            # ---- Uploaded file ----
            if file and file.filename != "":
                img = Image.open(file).convert("RGB")
                save_path = os.path.join("static/uploads", file.filename)
                img.save(save_path)
                image_path = url_for('static', filename=f"uploads/{file.filename}")

            # ---- URL input ----
            elif url and url != "":
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                image_path = url  # display directly from URL

            else:
                return render_template("index.html", prediction="No file or URL provided")

            # ---- Preprocess and predict ----
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                prediction = classes[preds.item()]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, filename=image_path)


# ---------------- Run Flask ----------------
if __name__ == "__main__":
    app.run(debug=True)
