import cv2
import numpy as np
import sys
import imutils
from imutils import perspective
import matplotlib.pyplot as plt

WIDTH=640
HEIGHT=480
ESC=27
def show(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class CrackDetector(object):
    def __init__(self, filename):
        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)
        self.img = cv2.imread(filename)
        self.origImg = np.copy(self.img)

    def gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def normalize(self):
        minVal = np.min(self.img)
        maxVal = np.max(self.img)
        self.img = (255*((self.img - minVal)/(maxVal-minVal))).astype("uint8")

    def medianBlur(self, kernelSize=3):
        self.img = cv2.medianBlur(self.img, kernelSize)

    def canny(self, thresholdMin=40, thresholdMax=120, show=False, winName='canny'):
        def maxOnChange(value):
            print("Max:", value)

        def minOnChange(value):
            print("Min:", value)

        minWidth = self.img.shape[0]/10
        minHeight = self.img.shape[1]/10


        if show:
            winName = 'canny'
            cv2.namedWindow(winName)
            cv2.createTrackbar('max', winName, 0, 255, maxOnChange)
            cv2.setTrackbarPos('max', winName, thresholdMax)
            cv2.createTrackbar('min', winName, 0, 255, minOnChange)
            cv2.setTrackbarPos('min', winName, thresholdMin)
            edges = cv2.Canny(self.img, thresholdMin, thresholdMax)
            cv2.imshow(winName, edges)

            print("Press <ESC> to destroy the window!")
            while True:
                if cv2.waitKey(20) & 0xFF == ESC:
                    break
                maxVal = cv2.getTrackbarPos('max', winName)
                minVal = cv2.getTrackbarPos('min', winName)
                edges = cv2.Canny(self.img, minVal, maxVal)
                cv2.imshow(winName, edges)
        else:
            edges = cv2.Canny(self.img, thresholdMin, thresholdMax)

        cv2.destroyAllWindows()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        self.img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        boxes = []
        #(_, contours, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, contours, _) = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (w, h) = cv2.boundingRect(c)[2:]
            if w < minWidth and h < minHeight:
                continue

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            boxes.append(box)
        return boxes

    def show(self, name=''):
        cv2.imshow(name, self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

detector = CrackDetector(sys.argv[1])
detector.show('original')

# convert image to GRAY format
detector.gray()
detector.show('gray')

# median blur
detector.medianBlur()
detector.show('median')

# normalize
detector.normalize()
detector.show('normalize')

# edge detection
boxes = detector.canny(show=True, winName='Crack detector')
detector.show('edge')

print(boxes)
output = np.copy(detector.origImg)
for b in boxes:
    output = cv2.rectangle(output, tuple(b[0]), tuple(b[2]), (0, 255, 0), 4)

show(output, 'output')

#ret, otsu = cv2.threshold(img, 20, 205, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#show(otsu, 'otsu')
