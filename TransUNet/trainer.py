import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import gc


def trainer_HuBMAP(args, model, snapshot_path):
    from datasets.dataset_HuBMAP import HuBMAP_dataset, RandomGenerator
    from datasets.dataset_HuBMAP import HuBMAP_dataset, Generator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = HuBMAP_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    ###azhe!
    db_val = HuBMAP_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                               transform=transforms.Compose(
                                   [Generator(output_size=[args.img_size, args.img_size])]))
    print("The length of val set is: {}".format(len(db_val)))
    ###azhe!

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)   
    ### val loader
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    low_val_loss_dice = np.inf 
    

    train_loss_list = []
    train_loss_dice_list = []
    val_loss_list = []
    val_loss_dice_list = []
    for epoch_num in range(max_epoch):
        total_train_loss = 0
        total_train_dice_loss = 0
        batch_num = 0
        for i_batch, sampled_batch in enumerate(trainloader):

            print("epoch: "+ str(epoch_num) + " training progress: {:.2f}".format(batch_num/len(trainloader)*100) + "%", end="\r")
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            ### get total loss
            total_train_loss += loss.item()
            total_train_dice_loss += loss_dice.item()
            ###  
            # update iter num for adaptive leraning rate
            # update batch_num for getting average loss
            writer.add_scalar('info/lr', lr_, iter_num)
            iter_num = iter_num + 1
            batch_num += 1

        avg_train_loss = total_train_loss/batch_num
        avg_train_loss_dice = total_train_dice_loss/batch_num
        writer.add_scalar('info/avg_train_loss', avg_train_loss, epoch_num)
        writer.add_scalar('info/avg_train_loss_dice', avg_train_loss_dice, epoch_num)
        train_loss_list.append(avg_train_loss)
        train_loss_dice_list.append(avg_train_loss_dice)
        np.save('train_loss.npy', train_loss_list)
        np.save('train_loss_dice.npy', train_loss_dice_list)


        if epoch_num % 1 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, epoch_num)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', outputs[1, ...] * 50, epoch_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, epoch_num)   


        ######################### VALIDATION ###########################
        total_val_loss = 0
        total_val_dice_loss = 0
        batch_num = 0

        for i_batch, sampled_batch in enumerate(valloader):
            print("epoch: "+ str(epoch_num) + " validation progress: {:.2f}".format(batch_num/len(valloader)*100) + "%", end="\r")
            model.eval()
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            
            np.save('val_pred.npy', outputs.detach().cpu().numpy())
            np.save('val_img.npy', image_batch.detach().cpu().numpy())
            np.save('val_label.npy',label_batch.detach().cpu().numpy())
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            
            ###
            total_val_loss += loss.item()
            total_val_dice_loss += loss_dice.item()
            ###
            
            batch_num = batch_num + 1
            
        avg_val_loss = total_val_loss/batch_num   
        avg_val_loss_dice = total_val_dice_loss/batch_num

        writer.add_scalar('info/avg_val_loss', avg_val_loss, epoch_num)
        writer.add_scalar('info/avg_val_loss_dice', avg_val_loss_dice, epoch_num)
        logging.info('Epoch %d : train_loss : %f, train_loss_dice: %f, val_loss: %f, val_loss_dice: %f' % (epoch_num, avg_train_loss, avg_train_loss_dice,avg_val_loss, avg_val_loss_dice))
        
        val_loss_list.append(avg_val_loss)
        val_loss_dice_list.append(avg_val_loss_dice)
        np.save('val_loss.npy', val_loss_list)
        np.save('val_loss_dice.npy', val_loss_dice_list)


        if epoch_num % 1 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('val/Image', image, epoch_num)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('val/Prediction', outputs[1, ...] * 50, epoch_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('val/GroundTruth', labs, epoch_num)  
        
        if avg_val_loss_dice < low_val_loss_dice:
            low_val_loss_dice = avg_val_loss_dice
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("current best model find!!")     

        del sampled_batch, image_batch, label_batch
        gc.collect()
        torch.cuda.empty_cache() 
        ###
        '''
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        '''
    writer.close()
    return "Training Finished!"


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_HuBMAP import HuBMAP_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = HuBMAP_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        iter_num = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"