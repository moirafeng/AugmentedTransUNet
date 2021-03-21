import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from tensorboardX import SummaryWriter

batch_size = 12

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/HuBMAP/test_512', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='HuBMAP', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_HuBMAP', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=batch_size, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()



def inference(args, model, test_save_path=None):
    from datasets.dataset_HuBMAP import HuBMAP_dataset,RandomGenerator
    from datasets.dataset_HuBMAP import HuBMAP_dataset, Generator
    db_test = HuBMAP_dataset(base_dir=args.root_path, split="test", list_dir=args.list_dir,transform=transforms.Compose(
                                   [Generator(output_size=[args.img_size, args.img_size])]))
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    ### Add validation here
    total_test_loss = 0
    total_test_dice_loss = 0
    batch_num = 0
    label_batch_sum = 0
    ce_loss = CrossEntropyLoss()
    num_classes = args.num_classes
    dice_loss = DiceLoss(num_classes)
    for i_batch, sampled_batch in enumerate(testloader):
        print(" testing progress: {:.2f}".format(batch_num/len(testloader)*100) + "%", end="\r")
        model.eval()
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        #print(label_batch.size())
        a = np.sum(label_batch.detach().cpu().numpy())
        print(a)
        outputs = model(image_batch)
        if a>label_batch_sum:
            label_batch_sum = a
            np.save('test_pred.npy', outputs.detach().cpu().numpy())
            np.save('test_img.npy', image_batch.detach().cpu().numpy())
            np.save('test_label.npy',label_batch.detach().cpu().numpy())
        
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice


        ###
        total_test_loss += loss.item()
        total_test_dice_loss += loss_dice.item()
        ###

        batch_num = batch_num + 1

    avg_test_loss = total_test_loss/batch_num   
    avg_test_loss_dice = total_test_dice_loss/batch_num
    print(avg_test_loss_dice)
    writer = SummaryWriter(snapshot_path + '/log')
    writer.add_scalar('info/avg_test_loss', avg_test_loss)
    writer.add_scalar('info/avg_test_loss_dice', avg_test_loss_dice)
    logging.info('test_loss : %f, test_loss_dice: %f' % (avg_test_loss, avg_test_loss_dice))    


    ###
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'HuBMAP': {
            
            'root_path': '../data/HuBMAP/test_512',
            'list_dir': './lists/lists_HuBMAP',
            'num_classes': 2,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(91)) ## 要改一下
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = '../predictions'
    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)