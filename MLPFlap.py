#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import skimage as skimage
from skimage import transform, color, exposure, io
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

GAME = 'bird'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2  # number of valid actions
SAVEPATH_MAIN="model.h5"
SAVEPATH_CRASH="crash.h5"
upframes=8
#que of fames
def addque(que, image):
    if len(que) > 29:
        que.remove(que[0])
    que.append(image)
    return que

#yeah...needed...
def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm

#this is to train the analyse accident model.(see readme)
def accident_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(6400,)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("We finish building the model")
    return model

def prepare_data_accidnet():
    pathflap = "trainingstuff/high/"
    pathdont = "trainingstuff/low/"
    dirflap = os.listdir(pathflap)
    dirdont = os.listdir(pathdont)
    flapnr = len(dirflap) + len(dirdont)

    x = np.zeros(shape=(flapnr, 6400), dtype=np.float32)
    y = np.zeros(shape=(flapnr, 2), dtype=np.float32)
    flapnr = 0
    for f in dirflap:
        image = skimage.io.imread(pathflap + f)
        image = transimage(image)
        x[flapnr, :] = image
        y[flapnr, :] = [1, 0]
        flapnr += 1
    for f in dirdont:
        image = skimage.io.imread(pathdont + f)
        image = transimage(image)
        x[flapnr, :] = image
        y[flapnr, :] = [0, 1]
        flapnr += 1
    return x, y


def train_accident(picsave):
    model = accident_model()
    x, y = prepare_data_accidnet()
    if picsave:
        model.fit(x, y, nb_epoch=650, batch_size=32)
    else:
        model.fit(x, y, nb_epoch=650, batch_size=16)
    return model

#main model...not even a lstm but a vanilla MLP of just 3 layers..
#Making this very easy and fast to train
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Dense(64, input_shape=(6400,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("We finish building the model")
    return model


#Get the screenshots from previous mistakes and make nice Numpy arrays of them
def prepare_data():
    global curtrainingfiles
    pathflap = "trainingstuff/flap/"
    pathdont = "trainingstuff/dont/"
    dirflap = os.listdir(pathflap)
    dirdont = os.listdir(pathdont)
    flapnr = len(dirflap) + len(dirdont)
    curtrainingfiles = flapnr
    x = np.zeros(shape=(flapnr, 6400), dtype=np.float32)
    y = np.zeros(shape=(flapnr, 2), dtype=np.float32)
    flapnr = 0
    for f in dirflap:
        image = skimage.io.imread(pathflap + f)
        image = transimage(image)
        x[flapnr, :] = image
        y[flapnr, :] = [1, 0]
        flapnr += 1
    for f in dirdont:
        image = skimage.io.imread(pathdont + f)
        image = transimage(image)
        x[flapnr, :] = image
        y[flapnr, :] = [0, 1]
        flapnr += 1
    return x, y

#train the model
def train(model):
    x, y = prepare_data()
    epoch = len(y)
    if epoch > 1500:
        epoch = 1500
    model.fit(x, y, nb_epoch=epoch, batch_size=16, verbose=1)
    return model

#resize, making black/white
def transimage(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (80, 80))
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    image = image.flatten()
    image = normalize(image)
    return image

#lets do it!
def playit(args, model, mega, model_crash, picsave):
    global games, totalframes, maxframes, curtrainingfiles,upframes
    game_state = game.GameState()
    #initialize actions
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    flap = np.zeros(ACTIONS)
    flap[1] = 1
    action = flap #set first action
    t = random.randint(1, 1000000) #random...so files saved for future training don't overlap
    imgque = [] #queue of recent frames. it is not an lstm so when crash because to high needs to export pic of last flap to train on
    count = 0
    flap_before_last = 0
    lastflap = 0
    while (True):
        count += 1

        ago = count - lastflap #last time we...eh...flapped
        preago = count - flap_before_last #second last time we flapped
        image_base, r_0, terminal = game_state.frame_step(action)
        #when crashed save the last picture if we crashed down, save the picture of the last time we flapped if we crash up
        if len(imgque) > 2:
            realterm = imgque[len(imgque) - 1]
        if terminal:
            if ago < 5:
                term = imgque[len(imgque) - (ago)]
            else:
                term = imgque[len(imgque) - 1]
            x = np.zeros(shape=(1, 6400), dtype=np.float32)
            x[0, :] = transimage(realterm)
            crash = model_crash.predict(x)[0]
            print("Crash prection", crash[0])#close to 1 if we crashed up, close to zero if we crashed at the bottom
            if (crash[0] >= 0.3) & (crash[0] < 0.7):
                if ago > upframes:
                    crash[0] = 0
                else:
                    crash[0] = 1
            if crash[0] <= 0.5:
                if preago <(2*upframes)+1:
                    print("gettings stuck...", preago) #i crashed below but just flapped..problem is the time before...This can only happen at the start sequence
                    term = imgque[len(imgque) - preago - ((2*upframes) - preago)]
                if ago < 10:
                    term = imgque[len(imgque) - ago - 2]

                if ago < 6:
                    term = imgque[len(imgque) - ago - (upframes - ago)]
                #save images for training!
                print("i think it was too low")
                if picsave:
                    skimage.io.imsave(
                        "trainingstuff/flap/ter_" + str(mega) + "_minaal" + str(ago) + "_" + str(t) + ".jpg", term)
                    skimage.io.imsave(
                        "trainingstuff/shouldbelow/ter_" + str(mega) + "l_minaal" + str(ago) + "_" + str(t) + ".jpg",
                        realterm)
            else:
                print("i think it was too high")
                if picsave:
                    skimage.io.imsave(
                        "trainingstuff/dont/last_" + str(mega) + "_minaal" + str(ago) + "_" + str(t) + ".jpg",
                        img_lastflap)
                    skimage.io.imsave(
                        "trainingstuff/shouldbehigh/ter_" + str(mega) + "h_minaal" + str(ago) + "_" + str(t) + ".jpg",
                        realterm)
            if count > maxframes:
                maxframes = count
            games += 1
            totalframes += count
            #flappy bird doesn't have scores...we measure success by frames played before crash
            print("last time flapped", ago)
            print("highest number of frames playyed without crash", maxframes)
            print("Latest number of frames", count)
            print("avarage frames", totalframes / games)

            break
        else:
            imgque = addque(imgque, image_base)
        #so...should we flap?
        image = transimage(image_base)
        x = np.zeros(shape=(1, 6400), dtype=np.float32)
        x[0, :] = image
        res = model.predict(x)
        if res[0][0] > 0.5:
            if ago > 8:
                action = flap
                flap_before_last = lastflap
                lastflap = count
                img_lastflap = image_base
            else:
                action = do_nothing

        else:
            action = do_nothing


def playGame(args):
    global games, totalframes, maxframes, curtrainingfiles,SAVEPATH_CRASH,SAVEPATH_MAIN
    picsave = True
    if os.path.exists(SAVEPATH_CRASH):
        model_crash=load_model(SAVEPATH_CRASH)
    else:
        model_crash = train_accident(picsave)
        model_crash.save(SAVEPATH_CRASH)
    curve = []
    iterati = 0

    while True:
        games = 0
        totalframes = 0
        maxframes = 0
        curtrainingfiles = 0

        if os.path.exists(SAVEPATH_MAIN):
            model = load_model(SAVEPATH_MAIN)
        else:
            model = buildmodel()
        if not args['mode'] == 'Run': # yes that means train...but easier to test in the editor if by no params it trains
            model = train(model)
            model.save(SAVEPATH_MAIN, overwrite=True)
        for t in range(10):
            playit(args, model, iterati, model_crash, picsave)
        iterati += 1
        curveitem = [iterati, maxframes, totalframes / games]
        curve.append(curveitem)
        my_df = pd.DataFrame(curve)
        my_df.to_csv('cur.csv', index=False, header=False)
        print("maxframe", maxframes)
        print("avarage frames", totalframes / games)
        print("Current training files", curtrainingfiles)
        print(curve)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=False)
    args = vars(parser.parse_args())
    playGame(args)


if __name__ == "__main__":
    main()
