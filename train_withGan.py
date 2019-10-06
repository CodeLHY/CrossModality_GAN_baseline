from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from ModelWithGAN import embed_net_withGAN
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=10, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path',default='logs/',type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int, metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=32, type=int, metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str, help='use GAN or not')
parser.add_argument('--Feature_GAN', default=0, type=int, metavar='m', help='method type')
parser.add_argument('--Loss_GAN', default="BCELoss", type=str, help="the loss function")
parser.add_argument('--drop', default=0.0, type=float, metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(0)

dataset = args.dataset
# --------------------------set data's path-------------------------
if dataset == 'sysu':
    data_path = '/home/hongyang/Data/sysu/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = 'RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
# ---------------------set checkpoint's directory-------------------
checkpoint_path = args.model_path
#  如果放checkpoint log 的文件夹不存在，那么就新建一个
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
# ----------------------set this training's name--------------------
suffix = ''
if args.Feature_GAN:
    suffix = 'GAN'
if args.method == 'id':
    suffix = suffix + dataset + '_id_bn_relu'
suffix = suffix + '_drop_{}'.format(args.drop)
suffix = suffix + '_lr_{:1.1e}'.format(args.lr)
suffix = suffix + '_dim_{}'.format(args.low_dim)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
suffix = suffix + '_' + args.arch

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)
# -------------------------GPU and things---------------------------
test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')
#  设置GPU/CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
feature_dim = args.low_dim
####################################################################
#
print('==> Loading data..')
# Data loading code
# --------------------------transforsm------------------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])
end = time.time()
# --------------------train、gall、query set-------------------------
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    #  todo: GenIdx再看看
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
# -----------------------------model-----------------------------
if args.Feature_GAN:
    net = embed_net_withGAN(args.low_dim, n_class, args.batch_size)
else:
    net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch)
net.to(device)
cudnn.benchmark = True
# --------------------resume with use checkpoint-----------------
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
# ------------------------loss and params------------------------
if args.method == 'id':
    criterion_id = nn.CrossEntropyLoss()
    criterion_id.to(device)
if args.Feature_GAN:
    if args.Loss_GAN == "MSELoss":
        criterion_GAN = nn.MSELoss(reduction="mean")  # 均方误差
        criterion_GAN.to(device)
    elif args.Loss_GAN == "BCELoss":
        # 由于输出不是0-1的 所以使用withlogits， witch pass the result to the sigmoid function at first, to make it belong to 0-1
        criterion_GAN = nn.BCEWithLogitsLoss(reduction="mean")  # 二分类的交叉熵误差
        criterion_GAN.to(device)
ignored_params = list(map(id, net.feature.parameters())) + list(map(id, net.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
# -------------------------set optimizer-------------------------
if args.optim == 'sgd':
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.feature.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
elif args.optim == 'adam':
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.feature.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}], weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 30:
        lr = args.lr
    elif epoch >= 30 and epoch < 60:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss_id = AverageMeter()
    train_loss_D = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    #  todo: here
    net.train()
    end = time.time()
    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        input1 = Variable(input1.cuda())  # RGB
        input2 = Variable(input2.cuda())  # IR
        labels = torch.cat((label1, label2), 0)
        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)
        outputs, feat, result_RGB_True, result_RGB_False, result_IR_True, result_IR_False = net(input1, input2)
        if args.method == 'id':
            loss_id = criterion_id(outputs, labels)
            loss = loss_id  # loss_id 是必要的！！！
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        # 如果使用GANloss 那么在epoch到了第十个的时候，特征稳定了，然后使用GANloss
        if args.Feature_GAN and epoch>=10:
            if batch_idx%5 == 0:

                # 每五个循环里面有四个是对D的更新，此时固定Generator的参数，并且要让D网络清醒地认识到什么是对的什么是错的，所以给他真实的标签
                loss_D_RGB_True = criterion_GAN(result_RGB_True, net.get_target_tensor(result_RGB_True, True).to(device))
                loss_D_RGB_False = criterion_GAN(result_RGB_False, net.get_target_tensor(result_RGB_False, False).to(device))
                loss_D_IR_True = criterion_GAN(result_IR_True, net.get_target_tensor(result_IR_True, True).to(device))
                loss_D_IR_False = criterion_GAN(result_IR_False, net.get_target_tensor(result_IR_False, False).to(device))
                loss_D = loss_D_RGB_True + loss_D_RGB_False + loss_D_IR_True + loss_D_IR_False
            else:
                # 每五个循环就要对Generater进行一次更新，这时，固定D的参数，并且使得生成的假的特征向真的特征靠拢(给他一个相反的label)
                loss_D_RGB_lie = criterion_GAN(result_RGB_False, net.get_target_tensor(result_RGB_False, True).to(device))
                loss_D_IR_lie = criterion_GAN(result_IR_False, net.get_target_tensor(result_IR_False, True).to(device))
                loss_D = loss_D_RGB_lie + loss_D_IR_lie
            loss = loss + loss_D
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_id.update(loss_id.item(), 2 * input1.size(0))
        if args.Feature_GAN and epoch>=10:
            train_loss_D.update(loss_D.item(), 2 * input1.size(0))
        train_loss_id.update(loss_id.item(), 2*input1.size(0))
        total += labels.size(0)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            if args.Feature_GAN:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'lr:{} '
                      'Loss_id: {train_loss_id.val:.4f} ({train_loss_id.avg:.4f}) '
                      'Loss_D: {train_loss_D.val:.4f} ({train_loss_D.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    data_time=data_time, train_loss_id=train_loss_id, train_loss_D=train_loss_D))
            else:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'lr:{} '
                      'Loss: {train_loss_id.val:.4f} ({train_loss_id.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    data_time=data_time, train_loss_id=train_loss_id))


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, args.low_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation mode
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, args.low_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # evaluation
    if dataset == 'regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc, mAP


# training
print('==> Start Training...')
for epoch in range(start_epoch, 61 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                              args.batch_size)
    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                                  drop_last=True)

    # training
    train(epoch)

    #  evaluation do not need to touch it
    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)
        # testing
        cmc, mAP = test(epoch)

        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], mAP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], mAP), file=test_log_file)
        test_log_file.flush()

        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model every 20 epochs
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))