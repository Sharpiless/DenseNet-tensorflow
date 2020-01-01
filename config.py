import os

if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

N_CLASSES = 16

BIAS = 32

GROWTH_RATE = 32

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 5e-4

BATCH_SIZE = 8

DATA_PATH = '../imagenet16/'

TRAIN_DATA_PATH = './dataset/train.pkl'

TEST_DATA_PATH = './dataset/test.pkl'

MAP_PATH = os.path.join(DATA_PATH, 'map.txt')

MODEL_PATH = './model'

MODEL_NAME = './model/model.ckpt'

TARGET_SIZE = 224

EPOCHES = 500

BATCHES = 128

KEEP_RATE = 0.85

CLASSES = ['knife', 'keyboard', 'elephant',
           'bicycle', 'airplane',  'clock',
           'oven', 'chair', 'bear', 'boat',
           'cat', 'bottle', 'truck', 'car',
           'bird', 'dog']
