import os
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import sys
# sys.path.append(r'C:\Users\Owner\Desktop\Hackathon\TFODCourse\tfod\Lib\site-packages\object_detection-0.1-py3.10.egg\object_detection\builders')
# import model_builder

# appending a path
# sys.path.append('\\tfod\\Lib\\site-packages\\object_detection-0.1-py3.10.egg\\object_detection\\builders\\model_builder.py')
# import model_builder
from object_detection.utils import config_util
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
import cv2 
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
IMAGE_PATH = 'cups.jpg'
img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

xd = detections.copy()
# function to count the number of objects detected in an image with a given confidence threshold of .8
# it takes in the detections dictionary and returns the number of objects detected dictionary
def count_objects(detections, threshold=0.8):
    a = {}
    # dictionary to store the number of detections
    a['num_detections'] = int(detections.pop('num_detections'))
    detections = {key: value[0, :a['num_detections']].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = a['num_detections']
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    a['detection_classes'] = detections['detection_classes']
    a['detection_scores'] = detections['detection_scores']
    count = 0
    for i in range(a['num_detections']):
        if a['detection_scores'][i] > threshold:
            count += 1
    return count

# function to count the number of objects detected of each class in an image with a given confidence threshold of .8
# it takes in the detections dictionary and returns the number of objects detected dictionary
def count_objects_each_class(detections, threshold=0.8):
    a = {}
    # dictionary to store the number of detections
    a['num_detections'] = int(detections.pop('num_detections'))
    detections = {key: value[0, :a['num_detections']].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = a['num_detections']
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    a['detection_classes'] = detections['detection_classes']
    a['detection_scores'] = detections['detection_scores']
    count = {}
    
    
    for i in range(a['num_detections']):
        if a['detection_scores'][i] > threshold:
            if a['detection_classes'][i] in count:
                count[a['detection_classes'][i]] += 1
            else:
                count[a['detection_classes'][i]] = 1
    return count


# function to rewrite the detection_counts with class names instead of class numbers and 0
def rewrite_detection_counts(count):
    # {1 : 3} to {'papercup': 3}
    # {'papercup':3} to {'papercup':3, 'bottle': 0, 'laptop': 0}
    nametags = {0 : 'violetbottle', 1 : 'papercup', 2 : 'laptop'}
    for i in range(3):
        if i not in count:
            count[i] = 0
    for i in count:
        count[nametags[i]] = count.pop(i)
    return count

    
            

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
# print(count_objects(xd))
print(rewrite_detection_counts(count_objects_each_class(xd)))