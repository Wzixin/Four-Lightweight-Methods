"""
模型蒸馏-简单代码示例
Step1：获得在Pruning环节，保存好的未经过任何操作的原始模型，作为教师模型保存在teacher_models文件夹下
       Top1=98.768
Step2：运行student_train，获得模型尺寸大大减小的student模型的原始性能
       Top1=90.535
Step3：运行distillation_train
       加载已保存的teacher_models，对学生网络知识蒸馏训练
       用Teacher_model输出的标签知识蒸馏Student_model，使student模型的性能提升
       Top1=92.768
"""
import argparse
import torch.utils
from utils import *
import torch.nn as nn
from MyModel import MyStudentNet
from MyModel import MyTeacherNet
import torch.utils.data.distributed
from torchvision import datasets, transforms

parser = argparse.ArgumentParser("MyNet")
parser.add_argument('--batch_size', type=int, default=256, help='每个批次大小：256')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数：10')
parser.add_argument('--learning_rate', type=float, default=0.01, help='初始化学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减率')
parser.add_argument('--temperature', type=float, default=2, help='标签软化温度 T')

parser.add_argument('--teacher_save', type=str, default='./teacher_models', help='已训练教师模型存放地址，取出')
parser.add_argument('--save', type=str, default='./Distill_student_models', help='已训练模型存放地址')
parser.add_argument('--data', default='/Users/mozixin/myData/', help='数据集存放地址 请更改为自己的地址')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='加载数据使用线程数量(default: 4)')

args = parser.parse_args()


def main():
    # 加载预定义模型
    student_model = MyStudentNet()
    teacher_model = MyTeacherNet()

    # 交叉墒损失函数
    criterion = nn.CrossEntropyLoss()
    # 软标签蒸馏 交叉墒损失函数
    Soft_criterion = CrossEntropy_SoftLabelloss()

    # 优化器 及 学习率更新策略
    optimizer = torch.optim.Adam(params=student_model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0., last_epoch=-1)

    start_epoch = 0
    best_top1_acc = 0

    # 加载已训练好的教师模型
    checkpoint_tar = os.path.join(args.teacher_save, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        print('loading teacher model checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        teacher_model.load_state_dict(checkpoint['state_dict'], strict=False)

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
        train_obj, train_top1_acc = train(epoch=epoch, train_loader=train_loader,
                                          student_model=student_model,
                                          teacher_model=teacher_model,
                                          criterion=Soft_criterion,
                                          optimizer=optimizer,
                                          temperature=args.temperature,
                                          scheduler=scheduler)
        # 测试
        valid_obj, valid_top1_acc = validate(epoch, val_loader, student_model, criterion)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        # 保存模型
        save_checkpoint(epoch, {
            'epoch': epoch,
            'state_dict': student_model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
            }, is_best, args.save)

        epoch += 1

    print("Best_top1_Acc = ", best_top1_acc)


def train(epoch, train_loader, student_model, teacher_model, criterion, optimizer, scheduler, temperature):
    losses = AverageMeter("loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    student_model.train()
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print("learning_rate:", cur_lr)

    for i, (images, target) in enumerate(train_loader):
        # 一个batch的图片数据 及 对应教师网络的输出标签
        images = images
        target = target

        with torch.no_grad():
            teacher_target = teacher_model(images)
            teacher_target = teacher_target / temperature
            teacher_target = torch.softmax(teacher_target, dim=1)

        # print(target[0])
        # print(teacher_target[0])  # 查看

        outputs = student_model(images)                                      # 计算模型输出预测值
        outputs = outputs / temperature

        loss = criterion(outputs, teacher_target.detach()) * (temperature**2)  # 计算损失loss
        prec1, prec5 = accuracy(outputs, target, topk=(1, 5))                # 计算准确率Top1 及 Top5

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
