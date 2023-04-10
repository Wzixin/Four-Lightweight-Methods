import argparse
import torch.utils
from utils import *
import torch.nn as nn
from MyModel import MyNet
import torch.utils.data.distributed
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

parser = argparse.ArgumentParser("MyNet")
parser.add_argument('--batch_size', type=int, default=256, help='每个批次大小：256')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数：10')
parser.add_argument('--save', type=str, default='./models', help='已训练模型存放地址')
parser.add_argument('--data', default='/Users/mozixin/myData/', help='数据集存放地址 请更改为自己的地址')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='加载数据使用线程数量(default: 4)')

parser.add_argument('--Threshold', type=float, default=0.1, help='剪枝阈值')

args = parser.parse_args()


def main():
    # 加载预定义模型
    model = MyNet()

    # 交叉墒损失函数
    criterion = nn.CrossEntropyLoss()

    # 如果有 加载已训练好模型权重
    checkpoint_tar = os.path.join(args.save, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        print('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 因为只评估性能，故只需加载测试集
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    # 测试剪枝前的模型性能
    # 本示例 Top1=98.768
    valid_obj, valid_top1_acc = validate(val_loader, model, criterion)
    print("剪枝之前 模型输出 top1_Acc = ", valid_top1_acc)
    # print("剪枝之前 第一个卷积层权重数据：", model.conv1.weights)

    print("---------------------------------进行剪枝-----------------------------------")

    # 对三个卷积层权重进行剪枝处理
    # 对于已保存模型的四个卷积层分别对应的权重
    # 简单剪枝策略：权重绝对值<剪枝阈值，那么会被剪去(置0)
    model.conv1.weights = Parameter(
        torch.where(torch.abs(model.conv1.weights) < args.Threshold, model.conv1.weights * 0, model.conv1.weights))
    model.conv2.weights = Parameter(
        torch.where(torch.abs(model.conv2.weights) < args.Threshold, model.conv2.weights * 0, model.conv2.weights))
    model.conv3.weights = Parameter(
        torch.where(torch.abs(model.conv3.weights) < args.Threshold, model.conv3.weights * 0, model.conv3.weights))

    # 测试剪枝后的模型性能
    valid_obj, valid_top1_acc = validate(val_loader, model, criterion)
    print("剪枝之后 模型输出 top1_Acc = ", valid_top1_acc)
    # print("剪枝之后 第一个卷积层权重数据：", model.conv1.weights)


def validate(val_loader, model, criterion):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [losses, top1],
        prefix='Test: ')

    model.eval()
    # 无梯度
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # 一个batch的图片数据 及 对应标签
            images = images
            target = target

            outputs = model(images)                                # 计算模型输出预测值
            loss = criterion(outputs, target)                      # 计算损失loss
            pred1, pred5 = accuracy(outputs, target, topk=(1, 5))  # 计算准确率Top1 及 Top5

            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)

            if i % 20 == 0:
                progress.display(i)

        print('***Test Acc @Top1 = {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
