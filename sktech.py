import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu" Load ControlNet model
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)
pipeline.to(device)

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    result = pipeline(image).images[0]
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    result.save(output_path)
    return output_path

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    output_path = process_image(file_path)
    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, port=5000)