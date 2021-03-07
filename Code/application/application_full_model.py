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
import time

import torch

from Code.models.heb_model import HebLetterToSentence
from Code.models.hybrid_model import HybridModel
from Code.utils.helpers import load_resnet_model, load_model
import numpy as np
import copy
from Code.application.utils import *
from Code.application.projectParams import *
import asyncio
from PIL import ImageDraw, Image
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from datetime import datetime

# Globals
#model = load_model(modelPath)
#model.load_weights(modelWeights)
# model = load_resnet_model()
# D:\Alon_temp\singlang\singLang_DLProg\out_puts\final_resnet_with_aug_colored_test_run_64.pt
# model = load_resnet_model("D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_test_run_64.pt")
# model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')
image_model = load_resnet_model('')

len_letters = 33
len_words = 52423
text_model = HebLetterToSentence(len_letters, 128, 512, 128, len_words, use_self_emmbed=True)
model = HybridModel(image_model,text_model)
path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\Code\\trianer\\full_run_test_pretrained_resnet\\resnet_full_run_test_50.pt"
model = load_model(model, path)
model.eval()

dataColor = (0, 255, 0)
pred = ''
prevPred = ''
sentence = ""
lastLetterWrote = ''
freq = 15
count = freq
threshold = 5
recording = None


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


async def record(roi):
    global recording,count

    if recording is None:
        recording = torch.tensor(roi).permute(2,0,1) / 255.
        print(recording)
        recording = recording.unsqueeze(0)
    else:
        now = datetime.now()
        print(now)
        new_image = torch.tensor(roi).permute(2,0,1) / 255.
        recording = torch.cat([recording,new_image.unsqueeze(0)])
        # count+=1



async def applay_on_recording():
    global recording,model
    model = model.cuda()
    # recording = recording.cat([recording[i] if i < recording.shape[0] else ])
    print(recording.shape)
    for i in range(recording.shape[0]):
        plt.imshow(recording[i].permute(1,2,0))
        plt.show()
    recording = recording.cuda()
    res,ht = model(recording.unsqueeze(0))
    res = res.squeeze(1)
    print(res)
    sm = torch.softmax(res,dim=1)
    print(sm)
    # print(torch.argmax(res,dim=1))
    recording = None



def main():
    """
    Main looping function.
    Apply pre-processing and asynchronously call for prediction.

    """
    global dataColor, window
    global count, pred,recording

    showMask = 0
    predict = 0
    count = 0
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
            if (count % 20 == 0):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(record(roi))
                if recording.shape[0] == 18:
                    predict = not predict
                    # if (predict):
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(applay_on_recording())
                    predict = not predict
            # count += 1


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
        if key == ord('r'): # record img
            # if (predict):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(record(roi))
            count+=1
        if key == ord('p'):  # mask
            # if (predict):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(applay_on_recording())
            # predict = not predict

        # time.sleep(1)

    cam.release()


if __name__ == '__main__':
    main()
