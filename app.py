import io
import os
from flask import Flask, Response, render_template, flash, request, redirect, url_for
import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename
from numpy import asarray
import numpy as np
import time


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_labels(filename):
    # with open(filename, 'r') as f:
    #     return [line.strip() for line in f.readlines()]
    with open('./annotations/instances_val2017.txt', 'r') as handle:
        annotate_json = json.load(handle)

    return [line.strip() for line in annotate_json.readlines()]


def create_figure():
    # launch predictor and run inference on an arbitrary image in the validation dataset
    graph_def_file = "ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_graph.pb"

    input_arrays = ["normalized_input_image_tensor"]
    output_arrays = ["raw_outputs/box_encodings", "raw_outputs/class_predictions"]
    input_shapes = {"normalized_input_image_tensor": [1, 320, 320, 3]}

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                                    input_arrays,
                                                                    output_arrays,
                                                                    input_shapes)
    tflite_model = converter.convert()

    with open('ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.tflite', 'wb') as f:
        f.write(tflite_model)

    image = 'static/2ca98d21a076b2ce.jpg'

    # launch predictor and run inference on an arbitrary image in the validation dataset
    # with Image.open(image) as img:
    #     (width, height) = (320, 320)
    #     img_resized = img.resize((width, height))
    #     img_array = asarray(img_resized, dtype=np.float32)  # first convert to a numpy array

    # Load the TFLite model and allocate tensors.
    interpreter = tf.compat.v1.lite.Interpreter(
        model_path="ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.tflite"
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    print(f'output_details: {output_details}')

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = tf.compat.v1.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels('./annotations/instances_val2017.txt')

    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    PIL_image = Image.fromarray(np.uint8(results)).convert('RGB')

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    print(f'results: {results}')
    print(f'results_shape: {results.shape}')

    # # load annotations to decode classification result
    # with open('./annotations/instances_val2017.txt', 'r') as handle:
    #     annotate_json = json.load(handle)
    # label_info = {idx + 1: cat['name'] for idx, cat in enumerate(annotate_json['categories'])}

    return PIL_image


@app.route('/')
def hello_world():
    figure = create_figure()
    output = io.BytesIO()
    FigureCanvas(figure).print_jpg(output)
    processed_image = Response(output.getvalue(), mimetype='image/jpeg')
    return render_template('index.html', processed_image=processed_image)


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))


if __name__ == "__main__":
    app.run(debug=True)
