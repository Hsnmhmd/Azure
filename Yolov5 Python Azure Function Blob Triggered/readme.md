# Blob Triggered Azure Function with YOLOV5 Inference 

## Description
This Azure function is triggered upon the creation of blobs (images) in the specified container. Once triggered, ONNX Yolov5 model detects 
the desired opjects and the function crops detected objects and saves them to another blob container.
The provided code does all required preprocessing for the inference

## Requirements and Usage
1- Exporting your best .pt yolov5 model to onnx format with batch size =1
2- python 3.9.12 runtime
3- provided requirement.txt
4- provide the path of the images and the connection string of the storage account in the function.json file
5- provide the required connection strings in the local.settings.json
6- modify the __init__.py file with your connection string, your class list and your model
