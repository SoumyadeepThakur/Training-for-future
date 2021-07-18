import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from transport import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

EPOCH = 200
BATCH_SIZE = 200
torch.set_num_threads(8)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(3)

def load_moons(root):

    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))

    return X, U, Y, indices

def init_weights(model):

    if type(model) == nn.Linear:
        nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0.01)

def plot_decision_boundary(c, u, X, Y, name):
    
    

    y = np.argmax(Y[u], -1)
    print(y)

    # Set min and max values and give it some padding
    x_min, x_max = -2.5, 2.0
    y_min, y_max = -2.0, 2.0
    h = 0.005
    # Generate a grid of points with distance h between them
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = torch.round(F.sigmoid(c(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u/11]]*900*800)).detach())).numpy()
    Z = Z.reshape(xx.shape)
    #Z = np.zeros_like(Z)
    # Plot the contour and training examples
    #sns.heatmap(Z)
    #plt.show()
    
    plt.title('%dth domain - %s' %(u, name))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, vmin=-1, vmax=2)
    plt.scatter(X[u][:, 0], X[u][:, 1], c=y, cmap=plt.cm.binary)
    plt.savefig('final_plots/%s_%f.pdf' %(name, u))

def plot_overlapping_boundary(c_1, c_2, u_1, u_2, X, Y, name):
        
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

    Y2 = np.argmax(Y[u_2], -1)
    Y1 = np.argmax(Y[u_1], -1)

    x_min, x_max = -2.5, 2.0
    y_min, y_max = -2.0, 2.0
    h = 0.005

    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z1 = c_2(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u_1/11]]*900*900)).detach().numpy()
    Z1 = Z1.reshape(xx.shape)
    Z2 = c_1(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]), torch.tensor([[u_2/11]]*900*900)).detach().numpy()
    Z2 = Z2.reshape(xx.shape)
    # Plot the contour and training examples
    #sns.heatmap(Z)
    #plt.show()
    #print(Z)
    #Z = (Z1 + 2*Z2)/3.0
    
    '''
    y1 = []
    y2 = []
    for i, x in enumerate(xx[0]):
        y = Z1[:,i]
        idx = np.where(y == 1.0)[0]
        y1.append(yy[:,0][int(np.min(idx))])
        y = Z2[:,i]
        idx = np.where(y == 1.0)[0]
        y2.append(yy[:,0][int(np.min(idx))])

    '''
    plt.xlabel(r'\textbf{feature} $x_1$')
    plt.ylabel(r'\textbf{feature} $x_2$')
    plt.xlim(-2.5, 2.0)
    plt.ylim(-2.0, 2.5)
        
    #plt.plot(xx[0], y1, 'c--', linewidth=3.0)
    #plt.plot(xx[0], y2, color='#00004c', linewidth=3.0)
    plt.contour(xx, yy, Z1, levels=[0], cmap=plt.cm.bwr, vmin=-1.0, vmax=2.0)
    plt.contour(xx, yy, Z2, levels=[0], cmap=plt.cm.seismic)
    prev = plt.scatter(X[u_2][:, 0], X[u_2][:, 1], s=25, c=Y2, cmap=plt.cm.seismic, alpha=0.7)
    cur = plt.scatter(X[u_1][:, 0], X[u_1][:, 1], s=25, c=Y1, cmap=plt.cm.bwr, vmin=-1.0, vmax=2.0, alpha=0.7)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('final_plots/%s_%f_%f.pdf' %(name, u_1, u_2))
    plt.clf()

class PredictionModelNN(nn.Module):

    
    def __init__(self, input_shape, hidden_shapes, output_shape, **kwargs):
        
        super(PredictionModelNN, self).__init__()

        self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
        self.leaky = kwargs['leaky']
        
        if self.time_conditioning:

            self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
            
        use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False
        self.regress = kwargs['task'] == 'regression' if kwargs.get('task') else False
        self.time_shape = 1

        if use_time2vec:
            self.time_shape = 8
            self.time2vec = Time2Vec(1, 8)
        else:
            self.time_shape = 1
            self.time2vec = None

        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()

        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        
        if len(self.hidden_shapes) == 0: # single layer NN, no TReLU

            self.layers.append(nn.Linear(input_shape, output_shape))
            self.relus.append(nn.LeakyReLU())

        else:

            self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
            if self.time_conditioning:
                self.relus.append(TimeReLU(data_shape=self.hidden_shapes[0], time_shape=self.time_shape, leaky=self.leaky))
            else:
                if self.leaky:
                    self.relus.append(nn.LeakyReLU())
                else:
                    self.relus.append(nn.ReLU())

            for i in range(len(self.hidden_shapes) - 1):

                self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
                if self.time_conditioning:
                    self.relus.append(TimeReLU(data_shape=self.hidden_shapes[i+1], time_shape=self.time_shape, leaky=self.leaky))
                else:
                    if self.leaky:
                        self.relus.append(nn.LeakyReLU())
                    else:
                        self.relus.append(nn.ReLU())


            self.layers.append(nn.Linear(self.hidden_shapes[-1], self.output_shape))

        self.apply(init_weights)

        for w in self.layers[0].parameters():
            print(w)
    
            
    def forward(self, X, times=None, logits=False, reps=False):

        if self.time_conditioning:
            X = torch.cat([X, times], dim=-1)

        if self.time2vec is not None:
            times = self.time2vec(times)

        #if self.time_conditioning:
        #    X = self.relus[0](self.layers[0](X), times)
        #else:
        #    X = self.relus[0](self.layers[0](X))

        for i in range(0, len(self.layers)-1):

            X = self.layers[i](X)
            if self.time_conditioning:
                X = self.relus[i](X, times)
            else:
                X = self.relus[i](X)

        X = self.layers[-1](X)
        #if self.regress:
        #   X = torch.relu(X)
        #else:
        #   X = torch.softmax(X,dim=1)
        '''
        if not logits:
            if self.output_shape > 1:
                X = F.softmax(X, dim=-1)
            else:
                X = F.sigmoid(X)
        '''
        return X        

def train_classifier(X, Y, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()
    
    Y_pred = torch.sigmoid(classifier(X))
    
    Y_true = torch.argmax(Y, 1).view(-1,1).float()

    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-15) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-15))
    
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

def train(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

    ## BASELINE 1- Sequential training with no adaptation ##

    X_source = X_data[source_indices[0]]
    Y_source = Y_data[source_indices[0]]
    print(Y_source)
    Y_source = np.array([0 if y[0] > y[1] else 1 for y in Y_source])
    print(Y_source.shape)
    X_aux = list(X_data[source_indices[1:]])
    Y_aux = list(Y_data[source_indices[1:]])
    Y_aux2 = []
    for i in range(len(Y_aux)):
        Y_aux2.append(np.array([0 if y[0] > y[1] else 1 for y in Y_aux[i]]))

    Y_aux = Y_aux2

    print(len(X_aux))
    print(len(Y_aux))

    X_target = X_data[target_indices[0]]
    Y_target = Y_data[target_indices[0]]
    Y_target = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

    X_source, X_aux, X_target = transform_samples_reg_otda(X_source, Y_source, X_aux, Y_aux, X_target, Y_target)

    print(X_source.shape)
    print(Y_source.shape)
    print(X_target.shape)
    print(Y_target.shape)
    print(X_aux[0].shape)
    print(Y_aux[0].shape)

    X_source = np.vstack([X_source] + X_aux)
    Y_source = np.hstack([Y_source] + Y_aux)
    Y_source = np.eye(2)[Y_source]
    Y_target = np.eye(2)[Y_target]
    print(X_source.shape)
    print(Y_source.shape)
    print(X_target.shape)
    print(Y_target.shape)
    
    classifier = PredictionModelNN(2, [50, 50], 1, leaky=True)
    classifier2 = PredictionModelNN(2, [50, 50], 1, leaky=True)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 5e-3)
    classifier_optimizer2 = torch.optim.Adam(classifier2.parameters(), 5e-3)


    past_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source).float(), torch.tensor(Y_source).float()),BATCH_SIZE,False)    
    print('------------------------------------------------------------------------------------------')
    print('TRAINING')
    print('------------------------------------------------------------------------------------------')
    for epoch in range(EPOCH):
        loss = 0
        for batch_X, batch_Y in past_data:
            loss += train_classifier(batch_X, batch_Y, classifier, classifier_optimizer)
        if epoch%10 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

    X_source = X_data[source_indices[0]]
    Y_source = Y_data[source_indices[0]]
    Y_source = np.array([0 if y[0] > y[1] else 1 for y in Y_source])
    X_aux = list(X_data[source_indices[1:6]])
    Y_aux = list(Y_data[source_indices[1:6]])
    Y_aux2 = []
    for i in range(len(Y_aux)):
        Y_aux2.append(np.array([0 if y[0] > y[1] else 1 for y in Y_aux[i]]))

    Y_aux = Y_aux2

    X_target = X_data[source_indices[6]]
    Y_target = Y_data[source_indices[7]]
    Y_target = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

    X_source, X_aux, X_target = transform_samples_reg_otda(X_source, Y_source, X_aux, Y_aux, X_target, Y_target)

    print(X_source.shape)
    print(Y_source.shape)
    print(X_target.shape)
    print(Y_target.shape)
    print(X_aux[0].shape)
    print(Y_aux[0].shape)

    X_source = np.vstack([X_source] + X_aux)
    Y_source = np.hstack([Y_source] + Y_aux)
    Y_source = np.eye(2)[Y_source]
    Y_target = np.eye(2)[Y_target]
    print(X_source.shape)
    print(Y_source.shape)
    print(X_target.shape)
    print(Y_target.shape)
    

    print('------------------------------------------------------------------------------------------')
    print('TRAINING')
    print('------------------------------------------------------------------------------------------')
    for epoch in range(EPOCH):
        loss = 0
        for batch_X, batch_Y in past_data:
            loss += train_classifier(batch_X, batch_Y, classifier2, classifier_optimizer2)
        if epoch%10 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

    print('------------------------------------------------------------------------------------------')
    print('TESTING')
    print('------------------------------------------------------------------------------------------')
        
    target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target).float(), torch.tensor(Y_target).float()),BATCH_SIZE,False)
    Y_pred = []
    for batch_X, batch_Y in target_dataset:
        batch_Y_pred = classifier2(batch_X).detach().cpu().numpy()

        Y_pred = Y_pred + [batch_Y_pred]  
    Y_pred = np.vstack(Y_pred)
    print('shape: ',Y_pred.shape)
    # print(Y_pred)
    Y_pred = np.array([0 if y < 0.5 else 1 for y in Y_pred])
    Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

    # print(Y_pred-Y_true)
    print(accuracy_score(Y_true, Y_pred))
    print(confusion_matrix(Y_true, Y_pred))
    print(classification_report(Y_true, Y_pred))    

    plot_overlapping_boundary(classifier, classifier2, 6, 9, X_data, Y_data, 'CDOT')
    
        
def main():

    X_data, U_data, Y_data, indices = load_moons('/home/sthakur/Training-for-future/data/Moons/processed')
    Y_data = np.eye(2)[Y_data]

    X_data = np.array([X_data[ids] for ids in indices])
    Y_data = np.array([Y_data[ids] for ids in indices])
    U_data = np.array([U_data[ids] for ids in indices])
    #X_data = preprocess_sleep2(X_data, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    train(X_data, Y_data, U_data, 11, [0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10])

if __name__ == "__main__":

    main()
