#!/usr/bin/env python3
"""
OpenCV integration with Keras.
Inspired by https://github.com/jgv7/CNN-HowManyFingers/blob/master/application.py

Controls:
use arrows to move the ROI box.
press p to turn prediction on/off.
press m to display/hide binary mask.
press esc to exit.

@author: Netanel Azoulay
@author: Roman Koifman
"""

#from keras.models import load_model
import torch
from Code.utils.helpers import load_resnet_model
import numpy as np
import copy
from Code.application.utils import *
from Code.application.projectParams import *
import asyncio
from PIL import ImageDraw, Image
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Globals
#model = load_model(modelPath)
#model.load_weights(modelWeights)
# model = load_resnet_model()
# D:\Alon_temp\singlang\singLang_DLProg\out_puts\final_resnet_with_aug_colored_test_run_64.pt
# model = load_resnet_model("D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_test_run_64.pt")
model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')
model.eval()
dataColor = (0, 255, 0)
pred = ''
prevPred = ''
sentence = ""
lastLetterWrote = ''
freq = 15
count = freq
threshold = 5


async def predictImg(roi):
    """
    Asynchronously prediction.

    :param roi: preprocessed image.
    """
    global count, sentence
    global pred, prevPred
    global lastLetterWrote

    img_tensor = torch.tensor(roi).permute(2,0,1) / 255

    x, y = model(img_tensor.unsqueeze(0))
    max = torch.argmax(y,dim=1)
    pred = convertEnglishToHebrewLetter(classes[max])
    if pred != prevPred:
        #print ("changed letter pred is {} but was {}".format(pred, prevPred))
        prevPred = pred
        count = 0
    elif count < threshold:
        count = count + 1
    elif lastLetterWrote != pred: #count == threshold and not didWritePred:
        # if pred == 'del':
        #     sentence = sentence[:-1]
        # else:
        #     sentence = sentence + pred
        # if pred == ' ':
        #     pred = 'space'
        sentence = sentence + pred
        lastLetterWrote = pred
        print(finalizeHebrewString(sentence))




def main():
    """
    Main looping function.
    Apply pre-processing and asynchronously call for prediction.

    """
    global dataColor, window
    global count, pred

    showMask = 0
    predict = 0
    fx, fy, fh = 10, 50, 45
    x0, y0, width = 400, 50, 224
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + width - 1, y0 + width - 1), dataColor, 12)

        # get region of interest
        roi = frame[y0:y0 + width, x0:x0 + width]
        # roi = binaryMask(roi)

        # apply processed roi in frame
        if showMask:
            img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            window[y0:y0 + width, x0:x0 + width] = img

        # take data or apply predictions on ROI
        if predict:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(predictImg(roi))

        if predict:
            dataColor = (0, 250, 0)
            cv2.putText(window, 'Strike'+'P'+'to start', (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, dataColor, 2, 1)
        else:
            dataColor = (0, 0, 250)
            cv2.putText(window, 'Prediction: OFF', (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, dataColor, 2, 1)

        # Add Letter prediction
        img_pil = Image.fromarray(window)
        draw = ImageDraw.Draw(img_pil)
        draw.text((fx, fy + fh), "Prediction: %s" % pred, font=font, fill=dataColor)
        draw.text((fx, fy + 2 * fh), 'Sample Timer: %d ' % count, font=font, fill=dataColor)
        window = np.array(img_pil)

        # Display
        cv2.imshow('Original', window)

        # Keyboard inputs
        key = cv2.waitKeyEx(10)

        # use ESC key to close the program
        if key & 0xff == 27:
            break

        elif key & 0xff == 255:  # nothing pressed
            continue

        # adjust the position of window
        elif key == 2490368:  # up
            y0 = max((y0 - 5, 0))
        elif key == 2621440:  # down
            y0 = min((y0 + 5, window.shape[0] - width))
        elif key == 2424832:  # left
            x0 = max((x0 - 5, 0))
        elif key == 2555904:  # right
            x0 = min((x0 + 5, window.shape[1] - width))

        key = key & 0xff
        if key == ord('m'):  # mask
            showMask = not showMask
        if key == ord('p'):  # mask
            predict = not predict

    cam.release()


if __name__ == '__main__':
    main()
