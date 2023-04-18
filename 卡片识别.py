#导入工具包
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import myutils
#设置参数
ap = argparse.ArgumentParser()
ap.add_argument("Ji", "--image", required=True,
help="path to input image")
ap.add_argument("-t", "--template", required=True,
help="path to template OCR-A image")
args=vars(ap.parse_args())
#指定信用卡类型
FIRST_NUMBER={
"3":"American Express",
"4":"Visa",
"5":"MasterCard",
"6":"Discover Card"
}
