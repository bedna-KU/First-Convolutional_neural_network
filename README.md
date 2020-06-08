# Your first Convolutional neural network with Keras
Example CNN from Wikipedia https://en.wikipedia.org/wiki/Convolutional_neural_network#/media/File:Typical_cnn.png
## Why
Blog in Slovak language https://linuxos.sk/blog/zumpa/detail/umela-inteligencia-prvy-prakticky-priklad-v-r/

## Installation
python3 -m pip install -r requirements.txt
git clone https://github.com/bedna-KU/First-Convolutional_neural_network.git
cd First-Convolutional_neural_network

## Files
    divide.py       - Divide images on train and test
    predict.py      - Predict single image
    predict_live.py - Predict images on desktop - live video
    train.py        - Train network

## Divide images to train and test
`python3 divide.py --test 20 --input mom --output data`

and

`python3 divide.py --test 20 --input no_mom --output data`

Create dataset with 80% train and 20% test images

    ├── data
       ├── test
       │   ├── mom
       │   │   ├── image_mom_1.jpg
       │   │   └── ...
       │   └── no_mom
       │       ├── image_no_mom_1.jpg
       │       └── ...
       └── train
           ├── mom
           │   ├── image_mom_1a.jpg
           │   └── ...
           └── no_mom
               ├── image_no_mom_1a.jpg
               └── ...

## Train convolutional neural network

`python3 train.py --input data`

**--input** directory with train and test images

## Predict single image
`python3 predict.py image.jpg`

**image.jpg** Image for prediction

## Predict images from desktop
python3 predict_live.py

![live predict keras](https://github.com/bedna-KU/First-Convolutional_neural_network/raw/master/live_predict_keras.gif)
