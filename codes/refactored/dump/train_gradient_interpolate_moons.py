import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from data_loaders import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FINETUNING_EPOCHS = 25
CLASSIFIER_EPOCHS = 30
FINETUNING_DOMAINS = 2

BATCH_SIZE = 100

SHUFFLE_BUFFER_SIZE=4096
ALPHA = 0.2

torch.set_num_threads(16)

class Time2Vec(nn.Module):

    '''
    Time encoding inspired by the Time2Vec paper
    '''

    def __init__(self, in_shape, out_shape):

        super(Time2Vec, self).__init__()
        linear_shape = out_shape//4
        dirac_shape = 0
        sine_shape = out_shape - linear_shape - dirac_shape
        self.model_0 = nn.Linear(in_shape, linear_shape)
        self.model_1 = nn.Linear(in_shape, sine_shape)
        #self.model_2 = nn.Linear(in_shape, dirac_shape)

    def forward(self, X):

        te_lin = self.model_0(X)
        te_sin = torch.sin(self.model_1(X))
        #te_dir = torch.max(10, torch.exp(-(self.model_2(X))))
        te = torch.cat([te_lin, te_sin], axis=1)
        return te

class TimeReLU(nn.Module):

    '''
    A ReLU with threshold and alpha as a function of domain indices.
    '''

    def __init__(self, data_shape, time_shape, leaky=False):
        
        super(TimeReLU,self).__init__()
        self.leaky = leaky
        self.model_0 = nn.Linear(time_shape, 16)
        
        self.model_1 = nn.Linear(16, data_shape)

        self.time_dim = time_shape        

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, 16)
            
            self.model_alpha_1 = nn.Linear(16, data_shape)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, X, times):

        thresholds = self.model_1(self.model_0(times))

        if self.leaky:
            alphas = self.model_alpha_1(self.model_alpha_0(times))
        else:
            alphas = 0.0
        X = torch.where(X>thresholds,X-thresholds,alphas*(X-thresholds))
        return X

'''
Prediction (classifier/regressor) model
Assumes 2 hidden layers
'''

class PredictionModel(nn.Module):

    def __init__(self, data_shape, hidden_shape, out_shape, time2vec=False):
        
        super(PredictionModel,self).__init__()

        self.time_dim = 1
        self.using_t2v = False
        if time2vec:
            self.using_t2v = True
            self.time_dim = 16
            self.t2v = Time2Vec(1, self.time_dim)

        self.layer_0 = nn.Linear(data_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_0.weight)
        nn.init.zeros_(self.layer_0.bias)
        self.relu_0 = TimeReLU(hidden_shape, self.time_dim, True)

        self.layer_1 = nn.Linear(hidden_shape, hidden_shape)
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.zeros_(self.layer_1.bias)
        self.relu_1 = TimeReLU(hidden_shape, self.time_dim, True)

        self.layer_2 = nn.Linear(hidden_shape, out_shape)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.zeros_(self.layer_2.bias)

    def forward(self, X, times):
        
        X = torch.cat([X, times], dim=1)
        if self.using_t2v:
            times = self.t2v(times)
        X = self.relu_0(self.layer_0(X), times)
        X = self.relu_1(self.layer_1(X), times)
        #X = self.relu_2(self.layer_2(X), times)
        X = self.layer_2(X)

        return X

def visualize_classifier(c,u, X_data, Y_data,**kwargs):

    X_new = np.linspace(-2.5,2.5, 100)
    Y_new = np.linspace(-2.5,2.5, 100)
    data = np.array([(x,y) for x in X_new for y in Y_new])
    # print(len(data))
    Y_0 = F.sigmoid(c(torch.tensor(data).float(), torch.tensor([[u/11]]*10000).float()))

    dat = (data[(Y_0.detach().numpy()>=0.5)[:,0]])
    dat2 = (data[(Y_0.detach().numpy()<0.5)[:,0]])
    plt.scatter(dat[:,0],dat[:,1], c='y')
    plt.scatter(dat2[:,0],dat2[:,1], c='c')
    uu = round(u)
    plt.scatter(X_data[uu][Y_data[uu][:,0]>=0.5][:,0],X_data[uu][Y_data[uu][:,0]>=0.5][:,1],c='b')
    plt.scatter(X_data[uu][Y_data[uu][:,0]<0.5][:,0],X_data[uu][Y_data[uu][:,0]<0.5][:,1],c='r')

    plt.savefig('figs/adversarial_cl_2%f.png' %u)
    #plt.show()
    plt.clf()

def visualize_grads(c,u, X_data, Y_data,**kwargs):

    X_new = np.linspace(-2.5,2.5, 100)
    Y_new = np.linspace(-2.5,2.5, 100)
    data = np.array([(x,y) for x in X_new for y in Y_new])
    # print(len(data))
    #Y_0 = F.sigmoid(c(torch.tensor(data).float(), torch.tensor([[u/11]]*10000).float()))
    uu = torch.tensor([[u/11]]*10000).float()
    uu.requires_grad_(True)
    Y_0 = F.sigmoid(c(torch.tensor(data).float(), uu))
    doy_dot = torch.autograd.grad(Y_0, uu, grad_outputs=torch.ones_like(Y_0), retain_graph=True)[0]
    print(doy_dot.shape)
    heat = doy_dot.detach().cpu().numpy().reshape(100, 100)
    print(np.abs(heat))
    print('Min: ', np.min(np.abs(heat)))
    print('Max: ', np.max(np.abs(heat)))
    print('Mean: ', np.mean(np.abs(heat)))
    plt.imshow(np.abs(heat), cmap='hot', interpolation='nearest')
    plt.savefig('figs/adversarial_2%f.png' %u)
    plt.clf()
    

def train_classifier(X, U, A, Y, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()
    
    Y_pred = F.sigmoid(classifier(X, A))
    
    Y_true = torch.argmax(Y, 1).view(-1,1).float()
    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss
'''

Inputs
X: data, 
U: domain index
A: time information
Y: label

L2 regularization of gradient
'''

def train_greg(X, U, A, Y, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()

    A_grad = A.clone()
    A_grad.requires_grad_(True)
    Y_pred = classifier(X, A_grad)
    partial_Y_pred_t = torch.autograd.grad(Y_pred, A_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
    #print(partial_Y_pred_t)
    #print(partial_Y_pred_t)
    
    Y_pred = F.sigmoid(Y_pred)
    Y_true = torch.argmax(Y, 1).view(-1,1).float()

    c = 1e-2

    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9)) + c*torch.mean(partial_Y_pred_t**2)
    
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

def finetune(X, U, A, Y, delta, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()

    A_grad = A.clone() - delta
    A_grad.requires_grad_(True)
    Y_pred = classifier(X, A_grad)
    partial_Y_pred_t = torch.autograd.grad(Y_pred, A_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
    
    #print(partial_Y_pred_t)
    Y_pred = Y_pred + delta * partial_Y_pred_t
    Y_pred = F.sigmoid(Y_pred)
    Y_true = torch.argmax(Y, 1).view(-1,1).float()

    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

'''

Inputs
X: data, 
U: domain index
A: time information
Y: label

Adversarially chooses delta and performs gradient interpolation using Taylor expansion

'''


def adversarial_finetune(X, U, A, Y, delta, classifier, classifier_optimizer):
    
    '''
    THIS IS THE METHOD THAT USING
    '''

    classifier_optimizer.zero_grad()
    delta.requires_grad_(True)

    for ii in range(5):

        A_grad = A.clone() - delta
        A_grad.requires_grad_(True)
        Y_pred = classifier(X, A_grad)
        Y_true = torch.argmax(Y, 1).view(-1,1).float()
        partial_Y_pred_t = torch.autograd.grad(Y_pred, A_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
        Y_pred = Y_pred + delta * partial_Y_pred_t
        Y_pred = F.sigmoid(Y_pred)
        loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))

        partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        delta = delta + 0.05*partial_loss_delta
        #print('%d %f' %(ii, delta.clone().detach().numpy()))
    
    delta = delta.clamp(-0.5, 0.5)
    A_grad = A.clone() - delta
    A_grad.requires_grad_(True)
    Y_pred = classifier(X, A_grad)
    partial_Y_pred_t = torch.autograd.grad(Y_pred, A_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
    
    #print(partial_Y_pred_t)
    Y_pred = Y_pred + delta * partial_Y_pred_t
    Y_pred = F.sigmoid(Y_pred)
    Y_true = torch.argmax(Y, 1).view(-1,1).float()

    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss


def train(X_data, Y_data, U_data, A_data, num_indices, source_indices, target_indices):

    X_source = X_data[source_indices]
    Y_source = Y_data[source_indices]
    U_source = U_data[source_indices]
    A_source = A_data[source_indices]

    X_target = X_data[target_indices]
    Y_target = Y_data[target_indices]
    U_target = U_data[target_indices]
    A_target = A_data[target_indices]

    print(X_source.shape)
    classifier = PredictionModel(3, 6, 1, True)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-3)

    X_past = np.vstack(X_source)
    U_past = np.hstack(U_source)
    Y_past = np.vstack(Y_source)
    A_past = np.hstack(A_source)

    #writer = SummaryWriter(comment='{}'.format(time.time()))

    print(X_past.shape)
    print(Y_past.shape)
    print(U_past.shape)

    print('------------------------------------------------------------------------------------------')
    print('TRAINING CLASSIFIERS')
    print('------------------------------------------------------------------------------------------')
    
    class_step = 0
    for epoch in range(CLASSIFIER_EPOCHS):
        past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(U_past).float(),
        											torch.tensor(A_past).float(), torch.tensor(Y_past).float()), BATCH_SIZE, True)
        class_loss = 0
        for batch_X, batch_U, batch_A, batch_Y in past_dataset:

            batch_U = batch_U.view(-1,1)
            batch_A = batch_A.view(-1,1)

            l = train_classifier(batch_X, batch_U, batch_A, batch_Y, classifier, classifier_optimizer)
            class_step += 1
            class_loss += l
        print("Epoch %d Loss %f"%(epoch,class_loss))

    X_past = X_source[0]
    U_past = U_source[0]
    Y_past = Y_source[0]

    
    print('------------------------------------------------------------------------------------------')
    print('FINETUNING CLASSIFIER')
    print('------------------------------------------------------------------------------------------')

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 5e-4)

    #domains = np.random.randint(0, len(X_source), 30)
    domains = np.arange(len(X_source)-FINETUNING_DOMAINS, len(X_source))
    ii = 0
    for index in domains:

        print('Finetuning step %d Domain %d' %(ii, index))
        ii+=1
        print('------------------------------------------------------------------------------------------')
        for epoch in range(FINETUNING_EPOCHS):
            cur_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source[index]).float(), torch.tensor(Y_data[index]).float(), 
                torch.tensor(A_source[index]).float(), torch.tensor(U_source[index]).float()), BATCH_SIZE, True)

            loss = 0
            for batch_X, batch_Y, batch_A, batch_U in cur_dataset:

                batch_U = batch_U.view(-1,1)
                batch_A = batch_A.view(-1,1)
                delta = torch.FloatTensor(1,).uniform_(-0.2, 0.2)
                #delta = torch.tensor([0.0])
                loss += adversarial_finetune(batch_X, batch_U, batch_A, batch_Y, delta, classifier, classifier_optimizer)
                #loss += train_greg(batch_X, batch_U, batch_A, batch_Y, classifier, classifier_optimizer)
                
            print("Epoch %d Loss %f"%(epoch,loss))
    
    for i in range(len(X_target)):

        target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(U_target[i]).float(),
            torch.tensor(A_target[i]).float(), torch.tensor(Y_target[i]).float()), BATCH_SIZE, False)
        Y_pred = []
        Y_label = []
        for batch_X, batch_U, batch_A, batch_Y in target_dataset:

            batch_U = batch_U.view(-1,1)
            batch_A = batch_A.view(-1,1)
            batch_Y_pred = F.sigmoid(classifier(batch_X, batch_A)).detach().cpu().numpy()

            Y_pred = Y_pred + [batch_Y_pred]
            Y_label = Y_label + [batch_Y]

        Y_pred = np.vstack(Y_pred).reshape(-1,)
        Y_label = np.vstack(Y_label)

        
        Y_pred = np.array([0 if y < 0.5 else 1 for y in Y_pred])
        Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target[i]])

     
        print(accuracy_score(Y_true, Y_pred))
        print(confusion_matrix(Y_true, Y_pred))
        print(classification_report(Y_true, Y_pred))    
        #print(np.mean((Y_pred - Y_label)**2))
        #print(np.mean(abs(Y_pred - Y_label)))
        #print(np.mean(abs(Y_pred - Y_label)/Y_label))
    
    ttt = torch.tensor(A_data[:,0]).view(-1, 1).float()
    rm = classifier.relu_0
    #print(rm.model_1(rm.model_0(classifier.t2v(ttt))))
    #print(rm.model_alpha_1(rm.model_alpha_0(classifier.t2v(ttt))))

    '''

    CODE TO VISUALIZE CLASSIFICATION BOUNDARY AND GRADIENTS
    '''
    '''
    visualize_grads(classifier, 9-0.1, X_data, Y_data)
    visualize_grads(classifier, 9-0.2, X_data, Y_data)
    visualize_grads(classifier, 9-0.3, X_data, Y_data)
    visualize_grads(classifier, 9, X_data, Y_data)
    visualize_grads(classifier, 9+0.1, X_data, Y_data)
    visualize_grads(classifier, 9+0.2, X_data, Y_data)
    visualize_grads(classifier, 9+0.3, X_data, Y_data)
    visualize_classifier(classifier, 9, X_data, Y_data)
    visualize_classifier(classifier, 8, X_data, Y_data)
    
    
    #for u in range(num_indices):
        #visualize_classifier(classifier, u, X_data, Y_data)
        #visualize_classifier(classifier, u-0.1, X_data, Y_data)
        #visualize_classifier(classifier, u-0.2, X_data, Y_data)
        #visualize_classifier(classifier, u-0.3, X_data, Y_data)
        #visualize_classifier(classifier, u+0.1, X_data, Y_data)
        #visualize_classifier(classifier, u+0.2, X_data, Y_data)
        #visualize_classifier(classifier, u+0.3, X_data, Y_data)
    '''
    
if __name__ == "__main__":

    X_data, Y_data, U_data = load_moons(11)
    #X_data, Y_data, A_data, U_data = load_sleep2('shhs1-dataset-0.15.0.csv')
    train(X_data, Y_data, U_data, U_data, 11, [0,1, 2, 3, 4, 5, 6, 7, 8], [9, 10])
