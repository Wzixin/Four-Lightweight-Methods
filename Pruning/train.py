"""
模型剪枝-简单代码示例
Step1：运行train 获得没有经过任何操作的原始模型 Top1=98.768
Step2：运行Pruning 测试不同程度剪枝后 的 模型性能
"""
import argparse
import torch.utils
from utils import *
import torch.nn as nn
from MyModel import MyNet
import torch.utils.data.distributed
from torchvision import datasets, transforms

parser = argparse.ArgumentParser("MyNet")
parser.add_argument('--batch_size', type=int, default=256, help='每个批次大小：256')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数：10')
parser.add_argument('--learning_rate', type=float, default=0.01, help='初始化学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减率')
parser.add_argument('--save', type=str, default='./models', help='已训练模型存放地址')
parser.add_argument('--data', default='/Users/mozixin/myData/', help='数据集存放地址 请更改为自己的地址')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='加载数据使用线程数量(default: 4)')

args = parser.parse_args()


def main():
    # 加载预定义模型
    model = MyNet()

    # 交叉墒损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器 及 学习率更新策略
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0., last_epoch=-1)

    start_epoch = 0
    best_top1_acc = 0

    # 如果有 加载已训练好模型权重
    checkpoint_tar = os.path.join(args.save, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        print('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 加载训练集 加载测试集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.data, train=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    # 训练 及 测试
    epoch = start_epoch
    while epoch < args.epochs:
        # 训练
        train_obj, train_top1_acc = train(epoch=epoch, train_loader=train_loader, model=model, criterion=criterion,
                                          optimizer=optimizer, scheduler=scheduler)
        # 测试
        valid_obj, valid_top1_acc = validate(epoch, val_loader, model, criterion)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        # 保存模型
        save_checkpoint(epoch, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
            }, is_best, args.save)

        epoch += 1

    print("Best_top1_Acc = ", best_top1_acc)


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    losses = AverageMeter("loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print("learning_rate:", cur_lr)

    for i, (images, target) in enumerate(train_loader):
        # 一个batch的图片数据 及 对应标签
        images = images
        target = target

        outputs = model(images)                                # 计算模型输出预测值
        loss = criterion(outputs, target.detach())             # 计算损失loss
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))  # 计算准确率Top1 及 Top5

        n = images.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 损失向后传播
        optimizer.step()       # 更新网络参数

        if i % 100 == 0:
            progress.display(i)

    scheduler.step()           # 更新学习率

    return losses.avg, top1.avg


def validate(epoch, val_loader, model, criterion):
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
