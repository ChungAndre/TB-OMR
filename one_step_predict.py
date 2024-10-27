import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os 

# Suppress TensorFlow CPU instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




import argparse
import cv2
import numpy as np
import tensorflow as tf
import utils
import warnings
import logging
import os

# Argument parser setup
parser = argparse.ArgumentParser(description='End-to-end music symbol recognition from image.')
parser.add_argument('-image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

# Image Segmentation Function
def boundingbox(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Load and process the input image
image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
horizontal_projection = np.sum(binary_image == 0, axis=1)
line_threshold = np.mean(horizontal_projection) * 0.5
staff_lines = np.where(horizontal_projection > line_threshold)[0]

# Group staff lines to identify staves
line_groups, current_group = [], []
for i in range(len(staff_lines) - 1):
    if staff_lines[i + 1] - staff_lines[i] < 20:
        current_group.append(staff_lines[i])
    else:
        current_group.append(staff_lines[i])
        line_groups.append(current_group)
        current_group = []
if current_group:
    line_groups.append(current_group)

# Crop the lines based on grouped staff lines
cropped_lines = []
margin = 30
for group in line_groups:
    y_min = max(0, group[0] - margin)
    y_max = min(binary_image.shape[0], group[-1] + margin)
    cropped_line = binary_image[y_min:y_max, :]
    cropped_line = boundingbox(cropped_line)
    cropped_lines.append(cropped_line)

# Load model and vocabulary for prediction
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

with open(args.vocabulary, 'r') as dict_file:
    int2word = {idx: word for idx, word in enumerate(dict_file.read().splitlines())}

saver = tf.compat.v1.train.import_meta_graph(args.model)
saver.restore(sess, args.model[:-5])
graph = tf.compat.v1.get_default_graph()

# Retrieve model tensors
input_tensor = graph.get_tensor_by_name("model_input:0")
seq_len_tensor = graph.get_tensor_by_name("seq_lengths:0")
keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.compat.v1.get_collection("logits")[0]

WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len_tensor)

# Predict music symbols for each cropped line
for cropped_line in cropped_lines:
    image = utils.resize(cropped_line, HEIGHT)
    image = utils.normalize(image)
    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
    seq_lengths = [image.shape[2] / WIDTH_REDUCTION]

    prediction = sess.run(decoded, feed_dict={
        input_tensor: image,
        seq_len_tensor: seq_lengths,
        keep_prob_tensor: 1.0,
    })

    str_predictions = utils.sparse_tensor_to_strs(prediction)
    for w in str_predictions[0]:
        print(int2word[w], end=" ")
    print("\n")

sess.close()


#!python newpredict.py -image muse3.png -model CP_semantic/trained_semantic_model-14000.meta -vocabulary Data/vocabulary_semantic.txt