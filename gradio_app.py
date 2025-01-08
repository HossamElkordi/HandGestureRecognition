import warnings
warnings.filterwarnings('ignore')

import cv2
import time
import numpy as np
import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    inp_type = input_details['dtype']
    inp_index = input_details["index"]
    out_index = output_details["index"]
    
    model = {
        'interpreter': interpreter, 
        'inp_scale': input_scale, 
        'inp_zero': input_zero_point, 
        'inp_type': inp_type,
        'inp_index': inp_index,
        'out_scale': output_scale, 
        'out_zero': output_zero_point,
        'out_index': out_index
    }
    return model

model_names = {
    'model': '/home/h.elkordi/projects/EdgeML/Project/models/model.tflite', 
    'model_quantized': '/home/h.elkordi/projects/EdgeML/Project/models/quantized_model.tflite',
}

models = {}
for k, v in model_names.items():
    models[k] = load_model(v)

def preprocess(image):
    img = np.array(image, dtype=np.float32) / 255.0
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(img, axis=(0, -1))

def predict(classifier, image):
    input_data = image
    
    interpreter = classifier['interpreter']
    input_scale = classifier['inp_scale']
    input_type = classifier['inp_type']
    input_index = classifier['inp_index']
    input_zero_point = classifier['inp_zero']
    output_scale = classifier['out_scale']
    output_index = classifier['out_index']
    output_zero_point = classifier['out_zero']
    
    if (input_scale, input_zero_point) != (0.0, 0):
        input_data = input_data / input_scale + input_zero_point
        input_data = input_data.astype(input_type)
        
    interpreter.set_tensor(input_index, input_data)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    predictions = interpreter.get_tensor(output_index)

    if (output_scale, output_zero_point) != (0.0, 0):
        predictions = predictions.astype(np.float32)
        predictions = (predictions - output_zero_point) * output_scale
    return predictions, end - start


def visualize_probabilities(probabilities):
    classes = ['NONE', 'PAPER', 'ROCK', 'SCISSORS']
    probs = probabilities.tolist()[0]
    class_idx = np.argmax(np.array(probs))

    cmap = LinearSegmentedColormap.from_list("gradient", ["blue", "green", "yellow", "red"])
    colors = [cmap(p) for p in probs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(classes, probs, height=0.5, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")

    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')

    plt.tight_layout()
    return fig, classes[class_idx]

def process_image(image, model):
    image = preprocess(image)
    probabilities, inf_time = predict(models[model], image)
    fig, detection = visualize_probabilities(probabilities)
    return fig, detection, f"{inf_time * 1000} ms"

with gr.Blocks() as demo:
    gr.Markdown("## Rock, Papper, Scissors Recognition")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Select an Image")
            process_button = gr.Button("Process Image")
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(model_names.keys()),
                label="Select Model",
                value="model"
            )
            plot_output = gr.Plot(label="Class Probabilities")
            with gr.Row():
                class_output = gr.Textbox(label="Predicted Class")
                time_output = gr.Textbox(label="Inference Time")
    process_button.click(fn=process_image, inputs=[image_input, model_selector], outputs=[plot_output, class_output, time_output])

demo.launch(share=True)