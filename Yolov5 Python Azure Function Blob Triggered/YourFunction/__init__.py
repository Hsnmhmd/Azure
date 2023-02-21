import json
import logging
from pickle import TRUE
import torch
import onnxruntime
import numpy as np
from Body_Parts_Function.yolo_onnx_preprocessing_utils import preprocess
from Body_Parts_Function.yolo_onnx_preprocessing_utils import non_max_suppression, _convert_to_rcnn_output
import azure.functions as func
from PIL import Image
import urllib.request
import cv2
from azure.storage.blob import BlobServiceClient
import io

'''
This Azure function is triggered upon the creation
of blob storages in the specified container. Once triggered, yolov5 model detects 
the desired opject and the function crops detected objects and saves them to another 
blob container.
'''
MY_CONNECTION_STRING = "The Connection string for your storage account in which the blobs that will trigger the function will be stored"
onnx_model_path="./YourFunction/YourModel.onnx"
batch_size = 1                      #The batch size that your yolov5 model is exported to work on 
def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
                 
    try:
      session = onnxruntime.InferenceSession(onnx_model_path)
      logging.info("ONNX model loaded.. ")
    except Exception as e: 
      logging.info(f"Error loading ONNX file:{str(e)} ")


    sess_input = session.get_inputs()
    sess_output = session.get_outputs()
    logging.info(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")

    for idx, input_ in enumerate(range(len(sess_input))):
        input_name = sess_input[input_].name
        input_shape = sess_input[input_].shape
        input_type = sess_input[input_].type
        logging.info(f"{idx} Input name : { input_name }, Input shape : {input_shape},Input type  : {input_type}")

    for idx, output in enumerate(range(len(sess_output))):
        output_name = sess_output[output].name
        output_shape = sess_output[output].shape
        output_type = sess_output[output].type
        logging.info(f" {idx} Output name : {output_name}, Output shape : {output_shape}, Output type  : {output_type}")

    batch, channel, height_onnx, width_onnx = session.get_inputs()[0].shape


    image_files = [] # List of images to work with multi images batches
    
    req = urllib.request.urlopen(myblob.uri)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image_files.append(cv2.imdecode(arr, cv2.IMREAD_COLOR)) 
    img_processed_list = []
    pad_list = []
    for i in range(batch_size):
        #image loading and resizing to the desired size
        #In the resizing step, the longer dimension is resized to 640 and the shorter is 
        #resized to a factor of 640 that keeps the original image ratios and the rest of 640 
        #in case of the shorter dimension is padded with constant value.
        img_processed, pad = preprocess(myblob.uri, img_size=640,fromurl=True)
        img_processed_list.append(img_processed)
        pad_list.append(pad)
        logging.info(f'pad {pad}')
    if len(img_processed_list) > 1:
        img_data = np.concatenate(img_processed_list)
    elif len(img_processed_list) == 1:
        img_data = img_processed_list[0]
    else:
        img_data = None

    assert batch_size == img_data.shape[0]


    def get_predictions_from_ONNX(onnx_session,img_data):
        """perform predictions with ONNX Runtime
        :param onnx_session: onnx model session
        :type onnx_session: class InferenceSession
        :param img_data: pre-processed numpy image
        :type img_data: ndarray with shape 1xCxHxW
        :return: boxes, labels , scores 
        :rtype: list
        """
        sess_input = onnx_session.get_inputs()
        sess_output = onnx_session.get_outputs()
        # predict with ONNX Runtime
        output_names = [ output.name for output in sess_output]
        pred = onnx_session.run(output_names=output_names,\
                                                  input_feed={sess_input[0].name: img_data})
        return pred[0]

    result = get_predictions_from_ONNX(session, img_data)

    #List of classes your model is suppossed to detect
    classes=[ 'class1','class2', 'class3']
    result_final = non_max_suppression(
        torch.from_numpy(result),
        conf_thres=0.20,
        iou_thres=0.45)

    def _get_box_dims(image_shape, box):
        box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
        height, width = image_shape[0], image_shape[1]

        box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

        box_dims['topX'] = box_dims['topX'] * 1.0 / width
        box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
        box_dims['topY'] = box_dims['topY'] * 1.0 / height
        box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

        return box_dims

    def _get_prediction(label, image_shape, classes):
        
        boxes = np.array(label["boxes"])
        labels = np.array(label["labels"])
        labels = [label[0] for label in labels]
        scores = np.array(label["scores"])
        scores = [score[0] for score in scores]

        bounding_boxes = []
        for box, label_index, score in zip(boxes, labels, scores):
            box_dims = _get_box_dims(image_shape, box)

            box_record = {'box': box_dims,
                        'label': classes[label_index],
                        'score': score.item()}

            bounding_boxes.append(box_record)

        return bounding_boxes

    bounding_boxes_batch = []
    for result_i, pad in zip(result_final, pad_list):
        label, image_shape = _convert_to_rcnn_output(result_i, height_onnx, width_onnx, pad)
        bounding_boxes_batch.append(_get_prediction(label, image_shape, classes))
    logging.info(json.dumps(bounding_boxes_batch, indent=1))


    img_np = image_files[0]  # replace with desired image index
    image_boxes = bounding_boxes_batch[0]  # replace with desired image index
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    x, y = img.size
    logging.info(img.size)
    blob_service_client=BlobServiceClient.from_connection_string(MY_CONNECTION_STRING)
    # Crop and load the crops
    for detect,i in zip(image_boxes,range(len(image_boxes))):
        label = detect['label']
        box = detect['box']
        ymin, xmin, ymax, xmax =  box['topY'], box['topX'], box['bottomY'], box['bottomX']
        topleft_x, topleft_y = x * xmin, y * ymin
        width, height = x * (xmax - xmin), y * (ymax - ymin)
        box = (topleft_x,topleft_y,  topleft_x+width, topleft_y+height)
        img2 = img.crop(box)

        buf = io.BytesIO()
        img2.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        logging.info(img2)

        name="the_name_of_image.jpeg"
        blob_client = blob_service_client.get_blob_client(container="container2", blob=name)
        blob_client.upload_blob(byte_im)
        logging.info(f"the crops is in blob= {name} are in container body-parts")
