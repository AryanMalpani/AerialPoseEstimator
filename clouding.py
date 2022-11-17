import cv2
import numpy as np
import os
import sys
def gammaCorrection(src, filename,gamma):
    path = "/home/aryna/Documents/Sem 5/CV/CrossLoc/datasets/naturescape/test_drone_real/rgb/"
#     direc = str(filename)+"/"
#     path = os.path.join(parent, direc) 
#     os.mkdir(path)
    gammaMatrix = [gamma]
    for gamma in gammaMatrix:
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        file = path+str(filename)
        cv2.imwrite(file, cv2.LUT(src, table))

folder = "/home/aryna/Documents/Sem 5/CV/CrossLoc/datasets/naturescape/test_drone_real/rgb_og/"

gamma=int(sys.argv[1])/10
print(gamma)
for filename in os.listdir(folder):
    temp = os.path.join(folder,filename)
    if(temp.endswith('png')):
        img = cv2.imread(temp)
        gammaCorrection(img, filename,gamma)
