
import cv2
import numpy as np

class_file_path = 'yolov3.txt'
weight_file_path = 'yolov3.weights'
config_file_path = 'yolov3.cfg'

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


## read in the class definitions
classes = None
with open(class_file_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

## setup the network
net = cv2.dnn.readNet(weight_file_path, config_file_path)

video_capture = cv2.VideoCapture(0)
# set width
video_capture.set(3, 416)
# set height
video_capture.set(4, 416)

while True:
    # Capture frame-by-frame
    ret, image = video_capture.read()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:               
                class_ids.append(class_id)

    print('---')
    print(class_ids.count(0))
    print('---')

    f = open("tmp_count.txt", "w")
    f.write(str(class_ids.count(0)))
    f.close()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break



