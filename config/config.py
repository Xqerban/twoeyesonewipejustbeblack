import torch

class Config(object):
    # 数据集
    TRAIN_LABEL = 'data/label_train.json'
    IMG_DIR = 'data/image/train/'
    TRAIN_ANN = 'data/annotations/train.json'
    VAL_ANN = 'data/annotations/val.json'

    TRAIN_RATIO = 0.8 # 训练集占比
    
    # 网络参数 TODO
    NUM_CLASSES = 3
    LEARNING_RATE = 0.005
    BATCH_SIZE = 8
    EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
