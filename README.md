# 新项目
我认为这个基于torch的项目不可能成为最终的答卷，但是其中会涉及到一些很有用的修改，我们改完之后可以直接沿用到原先的代码里面。

### 注意！由于预训练模型过大，超过100MB，无法上传到github，所以请自行下载预训练模型，在根目录下新建一个文件夹"pretrained_model"，将其放在文件夹内。下载地址是：https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

##### 新项目基于torch自带的faster R-CNN，具体的模型是：
```python
model = fasterrcnn_resnet50_fpn(weights=None)
```
## 环境
我暂时没有整理详细的依赖，大致只需要下面这几个   
- torchvision   
- tqdm   
- torch

## 代码解释
### dataset
这部分的设计大致和原先相同，主要的不同在于对bbox的处理和对于一些异常数据的检测。需要注意的也只有这两个地方
#### bbox
```python
def resize_bbox(bbox, img_size):
    W, H = img_size

    # 设置目标尺寸
    new_size = Config.IMG_SIZE

    # 计算缩放比例
    scale_x = new_size / W
    scale_y = new_size / H

    bbox = torch.tensor([
        bbox[0] * scale_x,  # xmin 缩放
        bbox[1] * scale_y,  # ymin 缩放
        bbox[2] * scale_x,  # xmax 缩放
        bbox[3] * scale_y  # ymax 缩放
    ])
    return bbox
```
由于训练代码的需要，我们需要把所有图片转化为大小一致的张量，但是在转化过程中会导致bbox和转化后的图片不匹配的问题。
所以，我们需要根据图片的比例修改bbox的大小，让他重新匹配
#### 异常数据
在给出的数据集中，异常数据有两种，估计都是阿里云给出的干扰项。
- 无意义，没有给出范围bbox的图片
- 实际上没有出错，但是故意给出了错误的bbox的图片，错误大致是：范围为负，范围超出图片大小
```python
 for ann in annotations:
            if ann['category_id'] == self.target_class_id:  # 仅保留目标类
                bbox = ann['bbox']  # 例如 [x, y, width, height]
                if bbox[2] > img.size[0] or bbox[3] > img.size[1] or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                bbox = resize_bbox(bbox, img.size)
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # 转换为 [xmin, ymin, xmax, ymax]
```
这段读取标注的代码处理了以上的异常，首先在我们的数据集处理中，没有给出范围的图片经过预处理，已经将他的catagory_id置为1.
此处先检测catagory_id，由于我们只要识别一类对象，也就是说faster R-CNN的类只有两种：
1（要识别的错误，即前景），0（没有错误或无意义的区域，即背景）。所以，此处的target_class_id总是为1。
如果图片的标注中给出的标注不是1，那么这个循环被跳过，如果他的bbox非法，那么这个循环被跳过。一旦跳过循环，在代码中这个图片的bbox和label就暂时为空  
在接下来的函数中，我们处理这些有标注但是没有设置bbox的。
```python
def collate_fn(batch):
    images, targets = zip(*batch)

    # 处理空目标框的情况
    for i in range(len(targets)):
        if targets[i]['boxes'].shape[0] == 0:  # 如果没有目标框
            targets[i]['boxes'] = torch.zeros((0, 4))  # 使用空框
            targets[i]['labels'] = torch.tensor([0], dtype=torch.int64)
        else:
            targets[i]['labels'] = targets[i]['labels'].to(torch.int64)

            # 处理图像堆叠
    images = torch.stack([image for image in images], 0)

    return images, targets
```
这个函数目前的处理方式，是对于那些没有bbox，即boxes.shape[0] == 0的图片，创建一个空的bbox，然后将label设置为0   
其中，
```python
 targets[i]['boxes'] = torch.zeros((0, 4))
```
为之前没有bbox的图像创建一个空的bbox，他的形状为（0,4），但是为空（ps:不是全零向量，是空向量）。
```python
题外话：
这里我认为创建一个空向量不足以成为最终的方案，还是应该创建一个有内容的向量，这是下一步要做的内容，我会写进TODO
```
### model&train
在此，我沿用了先前的config，项目各种变量的配置大致和原先相同。目前我没有对模型做出任何改进。只提供了一个测试函数   
测试函数获取模型生成的目标框,然后比较IOU，暂时认为IOU大于等于0.5即为识别成功。   
如需运行，只需运行Main即可。
