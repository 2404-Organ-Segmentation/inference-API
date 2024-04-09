from flask import Flask, request, send_file
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureChannelFirstd,
)
import os
import tempfile
import shutil
import pipeline as pl

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('file')
    if uploaded_files:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filenames = []
        for file in uploaded_files:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            filenames.append(filename)
        return "Files uploaded successfully: " + ", ".join(filenames)
    else:
        return "No files provided"

@app.route('/run', methods=['POST'])
def run_inference():
    # Get inference model ID from request data
    model_id = request.form.get('model_id')
    
    if model_id is None:
        return "Model ID not provided", 400

    uploaded_files = os.listdir(UPLOAD_FOLDER)
    if uploaded_files:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        model_type = "UNETR" if model_id == '0' else "SWINUNETR"
        model_path = 'models/unetr.pth' if model_id == '0' else 'models/swinunetr.pth'

        pipeline = pl.Pipeline(model_type=model_type, modality=1, num_of_labels=14, model_path=model_path)
        transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode="bilinear",
                ),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )
        pipeline.inference(data_folder=UPLOAD_FOLDER, output_folder=OUTPUT_FOLDER, transforms=transforms)

        # Delete uploaded files after inference
        shutil.rmtree(UPLOAD_FOLDER)
        return "Inference done. Output files: " + ", ".join(os.listdir(OUTPUT_FOLDER))
    else:
        return "No files uploaded for inference"

@app.route('/download', methods=['GET'])
def download_files():
    output_files = os.listdir(OUTPUT_FOLDER)
    if output_files:
        zip_path = tempfile.mktemp(suffix='.zip')
        shutil.make_archive(zip_path[:-4], 'zip', OUTPUT_FOLDER)
        
        # Delete all files in the output directory after zipping
        for file in output_files:
            os.remove(os.path.join(OUTPUT_FOLDER, file))
        
        return send_file(zip_path, as_attachment=True)
    else:
        return "No output files available for download"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
