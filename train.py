import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from dataset.coco import build_data
from model.faster_rcnn_vgg16 import *
from utils.config import opt
from dataset.dataset import *
import os
from trainer import FasterRCNNTrainer
from tqdm import tqdm
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc


def eval(dataloader, faster_rcnn):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(
        enumerate(dataloader)
    ):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = eval_detection_voc(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        use_07_metric=True,
    )

    return result


def train(**kwargs):
    opt._parse(kwargs)

    train_ratio = opt.TRAIN_RATIO
    input_json = opt.TRAIN_LABEL
    train_dir = opt.TRAIN_IMG_DIR
    train_labels_path = opt.TRAIN_ANN
    val_labels_path = opt.VAL_ANN

    # 如果训练数据和验证数据不存在，则生成数据
    if not os.path.exists(train_labels_path) or not os.path.exists(val_labels_path):
        build_data(train_ratio, input_json, train_dir, train_labels_path, val_labels_path)

    print("load data...")
    train_dataset = Dataset(train_labels_path, train_dir)
    val_dataset = TestDataset(val_labels_path, train_dir)

    train_loader = DataLoader(train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,num_workers=opt.test_num_workers)

    model = FasterRCNNVGG16(n_fg_class=opt.NUM_CLASSES, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32])
    print("model construct completed")

    trainer = FasterRCNNTrainer(model).to(opt.DEVICE)
    if opt.load_path:
        trainer.load(opt.load_path)
        print("load pretrained model from %s" % opt.load_path)

    best_map = 0 # 用于评估模型性能，以便在训练结束时可以保存或加载表现最好的模型。
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(train_loader)):
            scale = at.scalar(scale)# 放缩因子转换为标量值
            img, bbox, label = (
                img.to(opt.DEVICE).float(),
                bbox_.to(opt.DEVICE),
                label_.to(opt.DEVICE),
            )

            trainer.train_step(img, bbox, label, scale)

            # 每 opt.plot_every 次迭代绘制
            if (ii + 1) % opt.plot_every == 0:
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace()

                # plot loss 绘制损失函数的变化
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(
                    ori_img_, at.tonumpy(bbox_[0]), at.tonumpy(label_[0])
                )
                trainer.vis.img("gt_img", gt_img)#将绘制的图像显示在名为 "gt_img" 的窗口中

                # plot predict bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
                    [ori_img_], visualize=True
                )
                pred_img = visdom_bbox( # 在图像上绘制预测的边界框
                    ori_img_,
                    at.tonumpy(_bboxes[0]),
                    at.tonumpy(_labels[0]).reshape(-1),
                    at.tonumpy(_scores[0]),
                )
                trainer.vis.img("pred_img", pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win="rpn_cm")
                # roi confusion matrix
                trainer.vis.img(
                    "roi_cm", at.totensor(trainer.roi_cm.conf, False).float()
                )

        eval_result = eval(val_loader, model)
        trainer.vis.plot("test_map", eval_result["map"])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]["lr"]
        log_info = "lr:{}, map:{},loss:{}".format(
            str(lr_), str(eval_result["map"]), str(trainer.get_meter_data())
        )
        trainer.vis.log(log_info)

        if eval_result["map"] > best_map:#每次计算出新的 mAP 值后，代码会检查这个值是否超过了 best_map
            best_map = eval_result["map"]
            timestr = time.strftime("%m%d%H%M")
            save_path = "checkpoints/fasterrcnn_%s" % timestr
            best_path = trainer.save(save_path=save_path)

        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
    

if __name__ == "__main__":
    train()
