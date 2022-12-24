import cv2
import os


input_path = ""
output_path = ""

for filename in os.listdir(input_path):
    if(filename.endswith('png')):
        file_final = filename.rstrip("png")
        cv2.imwrite(output_path + file_final + "jpg", cv2.imread(filename))