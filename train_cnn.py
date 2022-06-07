import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import argparse
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from orcdata import *

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='5shot_resnet18_sk_er', type=str)
parser.add_argument('--data-dir', default='oracle_fs/img/oracle_200_5_shot', type=str)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--scale', default=0.8, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--wd', default=0., type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--device', type=int, nargs='+', default=[6])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(x) for x in args.device])
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
if args.device != 'cpu':
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load data
dataloaders = get_loader(args)
trainloader, testloader = dataloaders['train'], dataloaders['test']

# build model
model = torchvision.models.resnet152(num_classes=200)
model = nn.DataParallel(model)
model.cuda()

# train
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(tqdm(trainloader, ncols=0)):
        model.train()

        X, y = X.cuda(), y.cuda()

        # to use cuda
        X = Erode()(X)
        X = transforms.Normalize([0.84, 0.84, 0.84], [0.32, 0.32, 0.32])(X)

        lr = opt.state_dict()['param_groups'][0]['lr']

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    model.eval()
    test_loss, test_acc, m = 0, 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()

            with torch.cuda.amp.autocast():
                output = model(X)
            loss = criterion(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    train_acc_list.append(train_acc / n)
    train_loss_list.append(train_loss / n)
    test_acc_list.append(test_acc / m)
    test_loss_list.append(test_loss / m)

    print(
        f'[{args.name}] Epoch: {epoch + 1} | Train Acc: {train_acc / n:.3f}, loss: {train_loss / n:.3f}, '
        f'Test Acc: {test_acc / m:.3f}')

torch.save(model.state_dict(), 'model/' + args.name + '.pth')

plt.figure()
plt.plot(range(args.epochs), train_acc_list, label='train acc')
plt.plot(range(args.epochs), test_acc_list, label='test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig('./figure/'+args.name+'_acc.jpg', dpi=500)

plt.figure()
plt.plot(range(args.epochs), train_loss_list, label='train loss')
plt.plot(range(args.epochs), test_loss_list, label='test loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
plt.savefig('./figure/'+args.name+'_loss.jpg', dpi=500)
