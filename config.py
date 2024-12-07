from pprint import pprint
import torch

# Default Configs for training
# Att: the config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'


class Config:
    # data
    TRAIN_LABEL = 'data/label_train.json'
    TRAIN_IMG_DIR = 'data/image/train/'
    TEST_IMG_DIR = 'data/image/val/'
    TRAIN_ANN = 'data/annotations/train.json'
    VAL_ANN = 'data/annotations/val.json'
    MODEL_PATH = './pretrained_model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

    TRAIN_RATIO = 0.8 # 训练集占比
    
    # 网络参数 TODO
    IMG_SIZE = 512
    NUM_CLASSES = 2
    LEARNING_RATE = 0.005
    BATCH_SIZE = 4
    EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VAL_EVERY_N_EPOCHS = 1

    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.0
    roi_sigma = 1.0

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = "faster-rcnn"  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = "voc"
    pretrained_model = "vgg16"

    # training
    epoch = 30

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = "/tmp/debugf"

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = "checkpoints/vgg16_caffe.pth"

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print("======user config========")
        pprint(self._state_dict())
        print("==========end============")

    def _state_dict(self):
        return {
            k: getattr(self, k)
            for k, _ in Config.__dict__.items()
            if not k.startswith("_")
        }


opt = Config()
