import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '')))
import numpy as np
from itertools import cycle
 
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from utils import weights_init
from torch.utils.data import DataLoader
from dataset import mi_imagedata, mi_collate_img, bag2instances, generate_batch, load_dataset
from scipy.io import loadmat
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
 
from model.mivae import mlmivae_supervised
 
def get_loss(model, bags, bag_index, bag_label):
    with torch.no_grad():
        elbo, auxiliary_y, reconstruction_proba, KL_zx, KL_zy = \
            model.loss_function(bags, bag_index, bag_label, 1000)
    return elbo, auxiliary_y, reconstruction_proba, KL_zx, KL_zy
 
def get_accuracy(model, bags, bag_idx, bag_label, instance_label):
    with torch.no_grad():
        pred_bags, pred_instance = model.classifier_bag(bags, bag_idx.cpu(), 0.5 , L = 50)
    bag_acc_bag = accuracy_score(bag_label.cpu(), pred_bags.cpu())
    instance_auc = roc_auc_score(instance_label.cpu(), pred_instance.cpu())
    instance_aucpr = average_precision_score(instance_label.cpu(), pred_instance.cpu())
    return bag_acc_bag, instance_auc,instance_aucpr
 
def training_procedure(FLAGS, input_dim, dataset,rand_state):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda') 
    
    train_set, test_set = train_test_split(dataset, test_size = 0.1,random_state = rand_state)
    train_set, val_set = train_test_split(train_set, test_size = 0.1,random_state = rand_state)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((123.68/255, 116.779/255, 103.939/255), (0.5,0.5, 0.5)),
      ])
    transform_test= transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((123.68/255, 116.779/255, 103.939/255), (0.5,0.5, 0.5)),
        ])
 
    train_data = mi_imagedata(train_set,  FLAGS.cuda, transformations = transform)
    dataloader = DataLoader(train_data, batch_size = FLAGS.batch_size, shuffle=True, num_workers = 0,  collate_fn=mi_collate_img)    
    test_data = mi_imagedata(test_set, FLAGS.cuda, transformations = transform_test)
    testloader = DataLoader(test_data, batch_size = test_data.__len__(), shuffle=False, num_workers = 0,  collate_fn=mi_collate_img)    
    val_data = mi_imagedata(val_set, FLAGS.cuda, transformations = transform)
    valloader = DataLoader(val_data, batch_size = val_data.__len__(), shuffle=False, num_workers = 0,  collate_fn=mi_collate_img)
    
    model = mlmivae_supervised(FLAGS).to(device)
    model.apply(weights_init)
    model.train()
    auto_encoder_optimizer = optim.AdamW(model.parameters(), lr=FLAGS.initial_learning_rate, weight_decay=FLAGS.weight_decay)
    
    best_y_acc = 0.
    best_y_auc = 0.
    best_loss = 1000000
    
    print("Start Training!")    
    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        elbo_epoch = 0
        recon_epoch = 0
        y_epoch = 0
        KL_ins_epoch = 0
        KL_bag_epoch = 0
        for (i, batch) in enumerate(dataloader):
            # print(i)
            bag, bag_idx, bag_label, instance_label = batch
            auto_encoder_optimizer.zero_grad()            
            elbo, class_y_loss, reconstruction_proba, KL_instance, KL_bag = \
                model.loss_function(bag.float().to(device), bag_idx.to(device), bag_label.to(device), epoch)
            elbo.backward()
            auto_encoder_optimizer.step()  
              
            elbo_epoch  += elbo
            recon_epoch += reconstruction_proba
            y_epoch += class_y_loss
            KL_ins_epoch += KL_instance
            KL_bag_epoch += KL_bag
        elbo_epoch = elbo_epoch / (train_data.__len__()/FLAGS.batch_size)
        recon_epoch = recon_epoch / (train_data.__len__()/FLAGS.batch_size)
        y_epoch = y_epoch / (train_data.__len__()/FLAGS.batch_size)
        KL_ins_epoch = KL_ins_epoch / (train_data.__len__()/FLAGS.batch_size)
        KL_bag_epoch = KL_bag_epoch / (train_data.__len__()/FLAGS.batch_size)
 
        test_bag, test_bag_idx, test_bag_label, test_instance_label = next(iter(testloader))
        epoch_test_acc, epoch_test_auc, epoch_test_aucpr = get_accuracy(model, test_bag.float(), test_bag_idx, test_bag_label, test_instance_label)
 
        val_bag, val_bag_idx,val_bag_label, val_instance_label = next(iter(valloader))
        epoch_val_acc, epoch_val_auc, epoch_val_aucpr = get_accuracy(model, val_bag.float(), val_bag_idx, val_bag_label, val_instance_label)
        elbo, val_auxiliary_y, val_recon, val_KL_zx, val_KL_zy = get_loss(model, val_bag.float(), val_bag_idx, val_bag_label)
        epoch_val_loss = val_recon + val_auxiliary_y

        loss_epoch = epoch_val_loss
        acc_epoch = epoch_test_acc
        if loss_epoch < best_loss:
            best_y_acc = acc_epoch
            best_loss = loss_epoch
            torch.save(model,'path to model')
        elif loss_epoch == best_loss:
            if acc_epoch > best_y_acc:
                best_y_acc = acc_epoch
                best_loss = loss_epoch
                torch.save(model, 'path to model')
        
        if ((epoch + 1) % 1 ==0):
            print('Epoch #' + str(epoch+1) + '..............................................')
            print("Train loss  {:.5f}, recon_loss {:.5f}, y_loss {:.5f}".format (elbo_epoch, recon_epoch,  y_epoch))
            print("Val loss  {:.5f}, recon_loss {:.5f}, y_loss {:.5f}".format (epoch_val_loss, val_recon,  val_auxiliary_y))
            print("Val ACC {:.3f}, Val AUC  {:.3f}, Val AUC-PR {:3f}".format (epoch_val_acc, epoch_val_auc, epoch_val_aucpr))
    
    model = torch.load('path to model')
    test_acc, test_auc, test_aucpr = get_accuracy(model, test_bag.float(), test_bag_idx, test_bag_label, test_instance_label)
    print("ACC test: {:.4f}, AUC test: {:.4f}, AUC-PR test: {:.4f}" \
                  .format(test_acc, test_auc, test_aucpr))
    return test_acc, test_auc, test_aucpr


if __name__ == '__main__':
    import argparse
    import imageio
    from sklearn.model_selection  import ParameterGrid
    param_grid = {'instance_dim': [64], 'bag_dim' : [64], 'hidden_layer': [4],
                      'hidden_dim': [512], 'aux_loss_multiplier_y': [10000], 'attention_dim': [128]}
    grid = ParameterGrid(param_grid) 
    for params in grid:
        print(params)
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
        parser.add_argument('--batch_size', type=int, default=100, help="batch size for training")
        parser.add_argument('--feature_dim', type=int, default=200, help="number of classes on which the data set trained")
        parser.add_argument('--num_classes', type=int, default=2, help="number of classes on which the data set trained")
        parser.add_argument('--initial_learning_rate', type=float, default=5e-4, help="starting learning rate")
        parser.add_argument("--weight-decay", default=5e-4, type=float)
        parser.add_argument('--instance_dim', type=int, default=params['instance_dim'], help="dimension of instance factor latent space")
        parser.add_argument('--bag_dim', type=int, default=params['bag_dim'], help="dimension of bag factor latent space")
        parser.add_argument('--hidden_dim', type=int, default=params['hidden_dim'], help="dimension of hidden layers")
        parser.add_argument('--attention_dim', type=int, default=params['attention_dim'])
        parser.add_argument('--hidden_layer', type=int, default=params['hidden_layer'], help="number of hidden layers")
        parser.add_argument('--reconstruction_coef', type=float, default=1., help="coefficient for reconstruction term")
        parser.add_argument('--kl_divergence_coef', type=float, default=1., help="coefficient for instance KL-Divergence loss term")
        parser.add_argument('--kl_divergence_coef2', type=float, default=1., help="coefficient for bag KL-Divergence loss term")  
        parser.add_argument('--aux_loss_multiplier_y', type=float, default=params['aux_loss_multiplier_y'])
        parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
        parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")
        parser.add_argument('-w', '--warmup', type=int, default=0, metavar='N',
                            help='number of epochs for warm-up. Set to 0 to turn warmup off.')
        FLAGS = parser.parse_args(args=[])
 
        data_path = 'path to the Colon Cancer Data'
        file = open(data_path, 'rb')
        dataset = pickle.load(file)
        file.close()
        input_dim = (27,27,3)
 
        training_procedure(FLAGS, input_dim, dataset,i)
