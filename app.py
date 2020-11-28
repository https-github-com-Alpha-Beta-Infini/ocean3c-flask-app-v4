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

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_figure():
    # launch predictor and run inference on an arbitrary image in the validation dataset
    model_path = "yolo_v4_coco_saved_model"
    yolo_pred_cpu = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
    
    tflite_model = yolo_pred_cpu.convert()
    image_path = 'static/2ca98d21a076b2ce.jpg'
    with open(image_path, 'rb') as f:
        feeds = {'image': [f.read()]}
    results = tflite_model(feeds)

    # load annotations to decode classification result
    with open('annotations/instances_val2017.json') as f:
        annotate_json = json.load(f)
    label_info = {idx + 1: cat['name'] for idx, cat in enumerate(annotate_json['categories'])}

    # draw picture and bounding boxes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Image.open(image_path).convert('RGB'))
    wanted = results['scores'][0] > 0.1
    for xyxy, label_no_bg in zip(results['boxes'][0][wanted], results['classes'][0][wanted]):
        xywh = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        rect = patches.Rectangle((xywh[0], xywh[1]), xywh[2], xywh[3], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        rx = rx + rect.get_width() / 2.0
        ax.annotate(label_info[label_no_bg + 1], (rx, ry), color='w', backgroundcolor='g', fontsize=10,
                    ha='center', va='center', bbox=dict(boxstyle='square,pad=0.01', fc='g', ec='none', alpha=0.5))
    fig.savefig('uploads/plot.jpg')
    plt.show()
    return fig


@app.route('/')
def hello_world():
    figure = create_figure()
    output = io.BytesIO()
    FigureCanvas(figure).print_jpg(output)
    processed_image = Response(output.getvalue(), mimetype='image/jpeg')
    return render_template('index.html', processed_image=processed_image)


# @app.route('/classify')
# def plot_jpg():
#     figure = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(figure).print_jpg(output)
#     return Response(output.getvalue(), mimetype='image/jpeg')


# @app.route('/upload', methods = ['POST'])  
# def upload():  
#     if request.method == 'POST':  
#         file = request.files['file']  
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], "plot.jpg"))
#         return redirect("/")
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''


if __name__ == "__main__":
    app.run(debug=True)

