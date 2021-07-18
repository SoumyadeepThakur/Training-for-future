from easydict import EasyDict
import argparse
args = EasyDict()

args.epochs=100
args.dropout=0.0
args.lr=5e-3
args.gamma_exp=1000
args.hidden=800
args.ratio=1
args.dis_lambda=1.0
args.lambda_m=0.0
args.wgan='wgan'
args.clamp_lower=-0.15
args.clamp_upper=0.15
args.batch_size=200
args.num_train=100
args.loss='default'
args.evaluate=False
args.checkpoint='none'
args.save_head='tmp'
args.save_interval=20
args.log_interval=20
args.log_file='tmp_mlp'
seed=3
args.cuda=False


# from __future__ import print_function
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils_cida import plain_log
from utils_cida import write_pickle,read_pickle
from utils_cida import *
from utils_cida import gaussian_loss
from torch.utils import data
from data_loader import *
from sklearn.metrics import *
import matplotlib
import matplotlib.pyplot as plt


label_noise_std = 0.20
use_label_noise = False
use_inverse_weighted = True
discr_thres = 999.999
normalize = True
train_discr_step_tot = 2
train_discr_step_extra = 0
slow_lrD_decay = 1
norm = 12
fname_save = 'pred_tmp.pkl'
fname = '../../data/Moons/processed'
train_list = [0,1,2,3,4,5,6,7,8] #list(range(12))
mask_list =  [1]*11  #+ [0]           #[1] * 5 + [0] * 7
test_list =  [9, 10]             #list(range(12))

# torch.manual_seed(args.seed)



# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   """ Arguments: arg """
#   parser.add_argument('--seed',type=int)
    
#   args_p = parser.parse_args()
    
#   seed = int(args_p.seed)

if args.cuda:
    torch.cuda.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    Moonsdata(fname, train_list, normalize, mask_list),
    shuffle=True,
    batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(
    Moonsdata(fname, test_list, normalize),
    batch_size=args.batch_size, **kwargs)

print("Numpy version:", np.__version__)
print("Pytorch version:", torch.__version__)

class ClassificationDataSet(torch.utils.data.Dataset):
    
    def __init__(self, indices, transported_samples=None,target_bin=None, **kwargs):
        '''
        TODO: Handle OT
        Pass Transported samples as kwargs?
        '''
        self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
        self.transported_samples = transported_samples  # a 2-d array of OT maps
        
        self.root = kwargs['root_dir']
        self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
        self.transport_idx_func = kwargs['transport_idx_func'] if kwargs.get('transport_idx_func') else lambda x:x%1000
        self.num_bins = kwargs['num_bins'] if kwargs.get('num_bins') else 6
        self.base_bin = kwargs['num_bins'] if kwargs.get('num_bins') else 0   # Minimum whole number value of U
        #self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
        
        self.target_bin = target_bin
        self.X = np.load("{}/X.npy".format(self.root))
        self.Y = np.load("{}/Y.npy".format(self.root))
        self.A = np.load("{}/A.npy".format(self.root))
        self.U = np.load("{}/U.npy".format(self.root))
        self.drop_cols = kwargs['drop_cols_classifier'] if kwargs.get('drop_cols_classifier') else None
        
    def __getitem__(self,idx):

        index = self.indices[idx]
        data = torch.tensor(self.X[index]).float().to(self.device)   # Check if we need to reshape
        label = torch.tensor(self.Y[index]).long().to(self.device)
        auxiliary = torch.tensor(self.A[index]).float().to(self.device).view(-1, 1)
        domain = torch.tensor(self.U[index]).float().to(self.device).view(-1, 1)
        if self.transported_samples is not None:
            source_bin = int(np.round(domain.item() * self.num_bins)) # - self.base_bin
            # print(source_bin,self.target_bin)
            transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][self.transport_idx_func(idx)]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
            # print(source_bin,self.target_bin,transported_X.size())
            if self.drop_cols is not None:
                return data[:self.drop_cols],transported_X[:self.drop_cols], auxiliary, domain,  label
            return data,transported_X, auxiliary, domain,  label

        if self.drop_cols is not None:
            return data[:self.drop_cols], auxiliary, domain, label
        return data, auxiliary, domain, label

    def __len__(self):
        return len(self.indices)


class DomainEnc(nn.Module):
    def __init__(self):
        super(DomainEnc, self).__init__()
        
        self.fc1 = nn.Linear(2, 50)
        
        self.fc2 = nn.Linear(60, 60)

        self.fc3 = nn.Linear(60, 60)

        self.fc_final = nn.Linear(60, 20)
        

        self.fc1_var = nn.Linear(1, 10)
        

    def forward(self, x):
        x, domain = x

        # side branch for variable FC
        domain = domain.unsqueeze(1)
        #print(domain.shape)
        x_domain = F.relu(self.fc1_var(domain))

        # main branch
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)

        # combine feature in the middle
        x = torch.cat((x, x_domain), dim=1)
        x = F.relu(self.fc3(F.relu(self.fc2(x))))
        #x = self.drop2(x)

        # continue main branch
        x = F.relu(self.fc_final(x))

        return x

# Predictor
class DomainPred(nn.Module):
    def __init__(self):
        super(DomainPred, self).__init__()

        self.fc1 = nn.Linear(20, 20)

        self.fc_final = nn.Linear(20, 2)

    def forward(self, x):
        x, domain = x

        x = F.relu(self.fc1(x))

        x = self.fc_final(x)
        x = F.log_softmax(x)
        return x


# Discriminator: with BN layers after each FC, dual output
class DomainDDisc(nn.Module):
    def __init__(self):
        super(DomainDDisc, self).__init__()
        self.hidden = 20
        self.dropout = 0.3

        self.drop2 = nn.Dropout(self.dropout)

        self.fc3_m = nn.Linear(self.hidden, self.hidden)
        self.bn3_m = nn.BatchNorm1d(self.hidden)
        self.drop3_m = nn.Dropout(self.dropout)

        self.fc3_s = nn.Linear(self.hidden, self.hidden)
        self.bn3_s = nn.BatchNorm1d(self.hidden)
        self.drop3_s = nn.Dropout(self.dropout)

        self.fc4_m = nn.Linear(self.hidden, self.hidden)
        self.bn4_m = nn.BatchNorm1d(self.hidden)
        self.drop4_m = nn.Dropout(self.dropout)

        self.fc4_s = nn.Linear(self.hidden, self.hidden)
        self.bn4_s = nn.BatchNorm1d(self.hidden)
        self.drop4_s = nn.Dropout(self.dropout)

        self.fc5_m = nn.Linear(self.hidden, self.hidden)
        self.bn5_m = nn.BatchNorm1d(self.hidden)
        self.drop5_m = nn.Dropout(self.dropout)

        self.fc5_s = nn.Linear(self.hidden, self.hidden)
        self.bn5_s = nn.BatchNorm1d(self.hidden)
        self.drop5_s = nn.Dropout(self.dropout)

        self.fc6_m = nn.Linear(self.hidden, self.hidden)
        self.bn6_m = nn.BatchNorm1d(self.hidden)
        self.drop6_m = nn.Dropout(self.dropout)

        self.fc6_s = nn.Linear(self.hidden, self.hidden)
        self.bn6_s = nn.BatchNorm1d(self.hidden)
        self.drop6_s = nn.Dropout(self.dropout)

        self.fc7_m = nn.Linear(self.hidden, self.hidden)
        self.bn7_m = nn.BatchNorm1d(self.hidden)
        self.drop7_m = nn.Dropout(self.dropout)

        self.fc7_s = nn.Linear(self.hidden, self.hidden)
        self.bn7_s = nn.BatchNorm1d(self.hidden)
        self.drop7_s = nn.Dropout(self.dropout)

        self.fc_final_m = nn.Linear(self.hidden, 1)
        self.fc_final_s = nn.Linear(self.hidden, 1)        

    def forward(self, x):
        x, domain = x
        domain = domain.unsqueeze(1) / norm

        x = self.drop2(x)

        x_m = F.relu(self.bn3_m(self.fc3_m(x)))
        x_s = F.relu(self.bn3_s(self.fc3_s(x)))

        x_m = F.relu(self.bn4_m(self.fc4_m(x)))
        x_s = F.relu(self.bn4_s(self.fc4_s(x)))

        x_m = F.relu(self.bn5_m(self.fc5_m(x)))
        x_s = F.relu(self.bn5_s(self.fc5_s(x)))

        x_m = self.fc_final_m(x_m)
        x_s = self.fc_final_s(x_s) # log sigma^2

        return (x_m, x_s)

# Create models
encoder = DomainEnc()
predictor = DomainPred()
discriminator = DomainDDisc()
models = [encoder, predictor, discriminator]
#if args.cuda:
#    for model in models:
#        model.cuda()




torch.autograd.set_detect_anomaly(True)

# Set up optimizer
opt_D = optim.Adam(discriminator.parameters(), lr = args.lr) # lr 
opt_non_D = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr = args.lr) # lr 
lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=opt_D, gamma=0.5 ** (1/(args.gamma_exp*(train_discr_step_extra+1)) * slow_lrD_decay))
lr_scheduler_non_D = lr_scheduler.ExponentialLR(optimizer=opt_non_D, gamma=0.5 ** (1/args.gamma_exp))

ind = list(range(args.batch_size))
ind_test = list(range(1000))
bce = nn.BCELoss()
mse = nn.MSELoss()

# Training loop
def train(epoch):
    for model in models:
        model.train()
    sum_discr_loss = 0
    sum_total_loss = 0
    sum_pred_loss = 0
    for batch_idx, data_tuple in tqdm(enumerate(train_loader)):
        # print(batch_idx)
        if args.cuda:
            data_tuple = tuple(ele.cuda() for ele in data_tuple)
        if normalize:
            data_raw, target, domain, data, mask = data_tuple
        else:
            data, target, domain, mask = data_tuple

        # FF encoder and predictor
        encoding = encoder((data, domain))
        prediction = predictor((encoding, domain))

        if use_label_noise:
            noise = (torch.randn(domain.size()).cuda() * label_noise_std).unsqueeze(1)

        # train discriminator
        train_discr_step = 0
        while args.dis_lambda > 0.0:
            train_discr_step += 1
            discr_pred_m, discr_pred_s = discriminator((encoding, domain))
            discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / norm, np.mean(train_list) / norm, norm)
            for model in models:
                model.zero_grad()
            discr_loss.backward(retain_graph=True)
            opt_D.step()

            # handle extra steps to train the discr's variance branch
            if train_discr_step_extra > 0:
                cur_extra_step = 0
                while True:
                    discr_pred_m, discr_pred_s = discriminator((encoding, domain))
                    discr_loss = gaussian_loss(discr_pred_m.detach(), discr_pred_s, domain.unsqueeze(1) / norm)
                    for model in models:
                        model.zero_grad()
                    discr_loss.backward(retain_graph=True)
                    opt_D.step()
                    cur_extra_step += 1
                    if cur_extra_step > train_discr_step_extra:
                        break

            if discr_loss.item() < 1.1 * discr_thres and train_discr_step >= train_discr_step_tot:
                sum_discr_loss += discr_loss.item()
                break

        # handle wgan
        if args.wgan == 'wgan':
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        # train encoder and predictor
        pred_loss = masked_cross_entropy(prediction, target, mask)     # TODO - Change to MSE
        discr_pred_m, discr_pred_s = discriminator((encoding, domain))
        ent_loss = 0

        discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / norm)
        total_loss = pred_loss - discr_loss * args.dis_lambda

        for model in models:
            model.zero_grad()
        total_loss.backward()
        opt_non_D.step()
        sum_pred_loss += pred_loss.item()
        sum_total_loss += total_loss.item()

    lr_scheduler_D.step()
    lr_scheduler_non_D.step()

    avg_discr_loss = sum_discr_loss / len(train_loader.dataset) * args.batch_size
    avg_pred_loss = sum_pred_loss / len(train_loader.dataset) * args.batch_size
    avg_total_loss = sum_total_loss / len(train_loader.dataset) * args.batch_size
    log_txt = 'Train Epoch {}: avg_discr_loss = {:.5f}, avg_pred_loss = {:.3f}, avg_total_loss = {:.3f}'.format(epoch, avg_discr_loss, avg_pred_loss, avg_total_loss)
    print(log_txt)
    plain_log(args.log_file,log_txt+'\n')
    if epoch % args.save_interval == 0 and epoch != 0:
        torch.save(encoder, '%s.model_enc' % args.save_head)
        torch.save(predictor, '%s.model_pred' % args.save_head)
        torch.save(discriminator, '%s.model_discr' % args.save_head)

# Testing loop
def test(log_file=None):
    for model in models:
        model.eval()
    test_loss = 0
    rmse_loss = 0
    correct = 0
    l_data = []
    l_label = []
    l_gt = []
    l_true = []
    l_encoding = []
    l_domain = []
    l_prob = []
    #for data, target, domain in test_loader:
    for data_tuple in test_loader:
        if args.cuda:
            data_tuple = tuple(ele.cuda() for ele in data_tuple)
        if normalize:
            data_raw, target, domain, data = data_tuple
        else:
            data, target, domain = data_tuple
            data_raw = data
        encoding = encoder((data, domain))
        prediction = predictor((encoding, domain))
        preds = torch.argmax(prediction, 1)
        l_label += list(preds.detach().cpu().numpy())
        l_true += list(target.long().clone().cpu().numpy())
        
    test_loss /= len(test_loader.dataset)
    #correct   /= len(test_loader.dataset)
    #print(l_label, l_true)
    acc = accuracy_score(l_true, l_label)
    print('Accuracy: ', acc)
    log_txt = 'Test set: Accuracy: {:.7f}'.format(acc)
    
    return test_loss, acc

def plot_decision_boundary(u, X, X_r, Y, name):
    
    diff = (X-X_r).mean(0)
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

    matplotlib.rc('font', **font)
    #print(X.shape)
    #print(Y.shape)
    y = Y
    # Set min and max values and give it some padding
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    h = 0.005
    # Generate a grid of points with distance h between them
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    u_data = torch.FloatTensor([u]*1000*1000)
    print(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    print(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]) + diff)
    Z = encoder((torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]) + diff, u_data))
    Z = torch.argmax(torch.exp(predictor((Z, u_data))), -1)
    Z = Z.reshape(xx.shape)
    #sns.heatmap(Z)
    #print(Z.shape)
    #plt.show()
    # Plot the contour and training examples
    
    plt.title('%dth domain - %s' %(u, name))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, vmin=-1, vmax=2)
    plt.scatter(X_r[:, 0], X_r[:, 1], c=Y, cmap=plt.cm.binary)
    plt.savefig('%s_%f.png' %(name, u))

def plot_overlapping_boundary(u_1, u_2, X, X_r, X_r2, Y, Y_2, name):
        
    '''
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

    matplotlib.rc('font', **font)
    '''
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('font', family='serif', size=24, weight='bold')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amsfonts}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
    plt.rc('axes', linewidth=1)
    plt.rc('font', weight='bold')
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    # Set min and max values and give it some padding
    diff = (X-X_r).mean(0)
    x_min, x_max = -2.5, 2.0
    y_min, y_max = -2.0, 2.5
    h = 0.005
    # Generate a grid of points with distance h between them
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    u_data_1 = torch.FloatTensor([u_1]*900*900)
    u_data_2 = torch.FloatTensor([u_2]*900*900)

    Z1 = encoder((torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]) + diff, u_data_1))
    Z1 = torch.round(torch.exp(predictor((Z1, u_data_1)))[:,1]).detach().cpu().numpy()
    Z1 = Z1.reshape(xx.shape)

    Z2 = encoder((torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]) + diff, u_data_2))
    Z2 = torch.round(torch.exp(predictor((Z2, u_data_2)))[:,1]).detach().cpu().numpy()
    Z2 = Z2.reshape(xx.shape)

    plt.xlim(-2.5, 2.0)
    plt.ylim(-2.0, 2.5)

    plt.xlabel(r'\textbf{feature} $x_1$')
    plt.ylabel(r'\textbf{feature} $x_2$')
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)    
    #plt.plot(xx[0], y1, 'c--', linewidth=3.0)
    #plt.plot(xx[0], y2, color='#00004c', linewidth=3.0)
    

    plt.contour(xx, yy, Z1, levels=[0], cmap=plt.cm.bwr, vmin=-1.0, vmax=2.0)
    plt.contour(xx, yy, Z2, levels=[0], cmap=plt.cm.seismic)
    prev = plt.scatter(X_r[:, 0], X_r[:, 1], s=25, c=Y, cmap=plt.cm.seismic, alpha=0.7)
    cur = plt.scatter(X_r2[:, 0], X_r2[:, 1], s=25, c=Y_2, cmap=plt.cm.bwr, vmin=-1, vmax=2, alpha=0.7)
    plt.savefig('final_plots/%s_%f_%f.pdf' %(name, u_1, u_2))

    plt.clf()
    
    

def visualize_trajectory(encoder,predictor,indices,filename=''):
    td = ClassificationDataSet(indices=indices,root_dir=fname,device="cuda:0")
    fig, ax = plt.subplots(3, 3)
    ds = iter(torch.utils.data.DataLoader(td,1,False))
    for i in range(3):
        for j in range(3):
            x,a,u,y = next(ds)
            x_ = []
            y_ = []
            y__ = []
            y___ = []
            actual_time = u.view(1).detach().cpu().numpy()
            for t in tqdm(np.arange(actual_time-0.2,actual_time+0.2,0.005)):
                x_.append(t)
                t = torch.tensor([t*12]).float().to(x.device)
                t.requires_grad_(True)
                delta = (x[0,-1]*12 - t).detach()
                encoding = encoder((x, t))
                y_pred = predictor((encoding, t))
                # y_pred = .classifier(torch.cat([x[:,:-2],x[:,-2].view(-1,1)-delta.view(-1,1), t.view(-1,1)],dim=1), t.view(-1,1)) # TODO change the second last feature also
                partial_Y_pred_t = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred))[0]
                y_.append(y_pred.item())
                y__.append(partial_Y_pred_t.item())
                y___.append((-partial_Y_pred_t*delta + y_pred).item())
                # TODO gradient addition business
            ax[i,j].plot(x_,y_)
            ax[i,j].plot(x_,y__)
            # ax[i,j].plot(x_,y___)
            ax[i,j].set_title("time-{}".format(actual_time))

            # print(x_,y_)
            ax[i,j].scatter(u.view(-1,1).detach().cpu().numpy(),y.view(-1,1).detach().cpu().numpy())
    plt.savefig('traj_{}.png'.format(filename))
    plt.close()
    

best_acc = 0

for ep in range(args.epochs):
    train(ep)
    if ep % 10 == 0:
        loss, acc = test()
        if acc > best_acc: best_acc = acc

print('Best accuracy: ', best_acc)

models = [encoder,predictor,discriminator]
log_file = open("moons_cida.txt","a")
print("Seed - {}".format(seed),file=log_file)
print("Acc: {}".format(best_acc),file=log_file)
log_file.close()


plot_loader = torch.utils.data.DataLoader(Moonsdata(fname, [9], normalize), batch_size=400, **kwargs)
print(plot_loader)
for raw, target, u, x in plot_loader:
    plot_loader2 = torch.utils.data.DataLoader(Moonsdata(fname, [6], normalize), batch_size=400, **kwargs)
    for raw2, target2, u2, x2 in plot_loader2:

        plot_overlapping_boundary(6, 9, x, raw, raw2, target, target2, 'PCIDA')




