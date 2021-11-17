import torch
import torch.nn as nn
from utils import map_bag_embeddings, reorder_y
from utils import   accumulate_group_evidence_sum
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from nn import FullyConnected
"""
Genreative (decoder):
    x <--- z_i <--- z_B <--- Y
    |                |    
   |                |
  |------------------               
Inference (encoder):
    x ---> z_i
    x ---> z_B
    z_i --> z_B
    z_B ~~~ Y (aux)
"""
# Decoder, generative
class decoder_y(nn.Module):
    # p(z_B|y)
    def __init__(self, instance_latent_dim, bag_latent_dim, feature_dim, hidden_dim, hidden_layer, num_classes = 2):
        super(decoder_y, self).__init__()        
        self.linear_model = FullyConnected([num_classes] +
                                    [hidden_dim] * (1),
                                    layer_activation = nn.ReLU(), 
                                    final_activation = nn.ReLU(), batch_normalization = False)        
        self.fc21 = nn.Sequential(nn.Linear(hidden_dim, bag_latent_dim))
        self.fc22 = nn.Sequential(nn.Linear(hidden_dim, bag_latent_dim))
        
    def forward(self, y):
        hidden = self.linear_model(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden)
        return zy_loc, zy_scale    
 
    
class decoder_x(nn.Module):
    # p(x| z_I, z_B)
    def __init__(self, instance_latent_dim, bag_latent_dim, feature_dim, hidden_dim, hidden_layer):
        super(decoder_x, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(instance_latent_dim + bag_latent_dim, feature_dim, bias= True), nn.ReLU(),
                                  nn.Linear(feature_dim, 512), nn.ReLU(),
                                  nn.Linear(512, 48*5*5)
                                  )
        self.up1 = nn.Upsample(10)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(48, 32, kernel_size=3, bias=True), 
                                  nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size = 4, bias=True), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, stride=1))

    def forward(self, instance_latent_space, bag_latent_space):
        x = torch.cat((instance_latent_space,bag_latent_space), dim = -1)
        hidden1 = self.fc1(x)
        hidden2 = hidden1.view(-1, 48, 5, 5)
        hidden2 = self.up1(hidden2)
        hidden3 = self.de1(hidden2)
        hidden3 = self.up2(hidden3)
        loc_img = self.de2(hidden3)
        loc_img = self.de3(loc_img)
        return loc_img

# Encoders, inference, variational approximation
# encode z_I first, then encode z_B
class encoder_x(nn.Module):
    # qzx z_I ~ x
    # Take an instance x and z_B as input, encode the instance level latent z_I
    def __init__(self, instance_latent_dim, bag_latent_dim, feature_dim, hidden_dim, hidden_layer):
        super(encoder_x, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.instance_mu = nn.Linear(in_features=48 * 5 * 5, out_features=instance_latent_dim, bias=True)
        self.instance_logvar = nn.Sequential(nn.Linear(in_features=48 * 5 * 5, 
                                                       out_features=instance_latent_dim, bias=True))
        
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.flatten(start_dim = 1)
        
        instance_latent_space_mu = self.instance_mu(H)
        instance_latent_space_logvar = self.instance_logvar(H)
        return instance_latent_space_mu, instance_latent_space_logvar


class encoder_y(nn.Module):
    # qzy
    # Take an instance x as input, encode the bag level latent z_B, 
    # z_B is later accumulated together for the same bag
    def __init__(self, instance_latent_dim, bag_latent_dim, feature_dim, hidden_dim, hidden_layer):
        super(encoder_y, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            # conv layer (depth from 3 --> 36), 4x4 kernels
            nn.Conv2d(3, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv layer (depth from 36 --> 48), 3x3 kernels
            nn.Conv2d(32, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.bag_mu = nn.Linear(in_features=48 * 5 * 5, out_features=bag_latent_dim, bias=True)
        self.bag_logvar = nn.Sequential(nn.Linear(in_features=48 * 5 * 5, 
                                                  out_features=bag_latent_dim, bias=True))
        
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x).flatten(start_dim = 1)
        
        bag_latent_space_mu = self.bag_mu(H)
        bag_latent_space_logvar = self.bag_logvar(H) 
        return bag_latent_space_mu, bag_latent_space_logvar

class auxiliary_y_fixed(nn.Module):
    def __init__(self, instance_latent_dim, bag_latent_dim, feature_dim, hidden_dim, hidden_layer, num_classes = 2):
        super(auxiliary_y_fixed, self).__init__()
        self.fc = nn.Sequential(nn.Linear(bag_latent_dim+instance_latent_dim, 512), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(512,512), nn.ReLU(),nn.Dropout(),
                                    nn.Linear(512,1))
    def forward(self, z_ins, z_bag, bag_idx, bag_latent_embeddings):
        z = torch.cat((z_ins,z_bag),1)
        loc = self.fc(z)
        bags = (bag_idx).unique()
        M = torch.zeros((z_bag.shape[0], 1))
        for iter_id, bag in enumerate(bags):
            bag_id = bag.item()
            instances_bag = bag_idx.eq(bag_id).nonzero().squeeze()
            if instances_bag.numel()>0:
                if instances_bag.numel()>1:
                    M[iter_id, :] = torch.max(loc[instances_bag])
                else:
                    M[iter_id, :] = loc[instances_bag]
        return M, M, M, loc

class mlmivae_supervised(nn.Module):
    def __init__(self, args):
        super(mlmivae_supervised, self).__init__()
        self.batch_size = args.batch_size
        self.cuda = args.cuda
        self.bag_latent_dim = args.bag_dim
        self.instance_latent_dim = args.instance_dim
        self.hidden_layer = args.hidden_layer
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_classes = args.num_classes
        self.warmup = args.warmup
        
        self.reconstruction_coef = args.reconstruction_coef
        self.kl_divergence_coef = args.kl_divergence_coef
        self.kl_divergence_coef2 = args.kl_divergence_coef2
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.attention_dim = args.attention_dim
        self.decoder_x= decoder_x(self.instance_latent_dim, self.bag_latent_dim, 
                                   self.feature_dim, self.hidden_dim, self.hidden_layer)
        self.decoder_y = decoder_y(self.instance_latent_dim, self.bag_latent_dim, 
                                   self.feature_dim, self.hidden_dim, self.hidden_layer)
        self.encoder_x = encoder_x(self.instance_latent_dim, self.bag_latent_dim, 
                                   self.feature_dim, self.hidden_dim, self.hidden_layer)
        self.encoder_y = encoder_y(self.instance_latent_dim, self.bag_latent_dim, 
                                   self.feature_dim, self.hidden_dim, self.hidden_layer)
        self.auxiliary_y_fixed = auxiliary_y_fixed(self.instance_latent_dim, self.bag_latent_dim, 
                                   self.feature_dim, self.hidden_dim, self.hidden_layer,
                                   self.num_classes)
    def forward(self, bag, bag_idx, bag_label):
        # Encode
        # encode instance latents
        instance_mu, instance_logvar = self.encoder_x(bag)
        instance_logvar = instance_logvar.mul(0.5).exp_()
        qzx = dist.Normal(instance_mu, instance_logvar)
        zx_q = qzx.rsample()  # [# of instances, instance_latent_dim]
        
        # encode bag latents
        bag_mu, bag_logvar = self.encoder_y(bag)
        bag_logvar = bag_logvar.mul(0.5).exp_()
        grouped_mu, grouped_logvar, list_g, sizes_group = accumulate_group_evidence_sum(
                bag_mu, bag_logvar, bag_idx, self.cuda)
        qzy = dist.Normal(grouped_mu, grouped_logvar)
        zy_q = qzy.rsample() # [# of bags, bag_latent_dim]    
        # use zy_q to generate bag_embeddings that corresponds to instance embeddings zx_q
        bag_latent_embeddings = map_bag_embeddings(bag, zy_q, bag_idx, list_g)
        n_groups = grouped_mu.size(0)

        #reorder by the same order as bag_latent_embeddings
        reordered_y = reorder_y(bag_label, bag_idx, list_g)
        one_hot = torch.nn.functional.one_hot(reordered_y.to(torch.long),2)
        zy_p_loc, zy_p_scale  = self.decoder_y(one_hot.float())
        zy_p_scale = zy_p_scale.mul(0.5).exp_()
        pzy = dist.Normal(zy_p_loc, zy_p_scale)
        KL_zy = (pzy.log_prob(zy_q) - qzy.log_prob(zy_q)).mean()

        # kl-divergence error for instance latent space
        KL_zx = - 0.5 * (instance_mu.pow(2) + instance_logvar.pow(2) - torch.log(instance_logvar.pow(2)) - 1).mean()

        # probablistic reconstruct samples
        x_recon = self.decoder_x(zx_q, bag_latent_embeddings).flatten(start_dim = 1)
        x_target = bag.flatten(start_dim = 1)
        loss = nn.MSELoss(reduction = 'mean')
        reconstruction_proba = loss(x_recon,x_target)
        
        y_hat, _,_,_ = self.auxiliary_y_fixed(zx_q, zy_q, bag_idx, bag_latent_embeddings)
        
        loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        auxiliary_loss_y= loss(y_hat.squeeze(), reordered_y)

        return reconstruction_proba, KL_zx, KL_zy, auxiliary_loss_y,  n_groups

    def loss_function(self, bag, bag_idx, bag_label, epoch):
        # supervised
        if self.warmup > 0:
            kl_divergence_coef = self.kl_divergence_coef
            kl_divergence_coef2 = self.kl_divergence_coef2
            if epoch > self.warmup:
                aux_loss_multiplier_y =  self.aux_loss_multiplier_y
            else:
                aux_loss_multiplier_y = 0
        else:
            kl_divergence_coef = self.kl_divergence_coef
            kl_divergence_coef2 = self.kl_divergence_coef2
            aux_loss_multiplier_y = self.aux_loss_multiplier_y
        reconstruction_proba, KL_zx, KL_zy, auxiliary_y, n_groups \
            = self.forward(bag, bag_idx, bag_label)
        
        reconstruction_proba = reconstruction_proba 
        KL_zx = KL_zx
        KL_zy = KL_zy
        auxiliary_y = auxiliary_y
        elbo = (  reconstruction_proba - kl_divergence_coef * KL_zx \
                - kl_divergence_coef2 * KL_zy + aux_loss_multiplier_y * auxiliary_y )
        return elbo, auxiliary_y, reconstruction_proba, KL_zx, KL_zy
        
    def classifier_bag(self, bag, bag_idx, threshold=0.5, L=10):
        with torch.no_grad():
            ins_loc, ins_scale = self.encoder_x.forward(bag)
            
            zy_q_loc, zy_q_scale = self.encoder_y.forward(bag)
            zy_q_scale = zy_q_scale.mul(0.5).exp_()
            bag_mu, bag_logvar, list_bag, sizes_bag = accumulate_group_evidence_sum(
                zy_q_loc, zy_q_scale, bag_idx, self.cuda)
            bag_latent_embeddings = map_bag_embeddings(bag, bag_mu, bag_idx, list_bag)
            
            Y_prob, _,_, attention = self.auxiliary_y_fixed(ins_loc, bag_mu, bag_idx, bag_latent_embeddings)
            Y_prob = torch.sigmoid(Y_prob)
            Y_hat= torch.ge(Y_prob, 0.5).float()
        return Y_hat, attention
