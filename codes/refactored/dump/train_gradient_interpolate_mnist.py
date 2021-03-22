import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from data_loaders import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FINETUNING_EPOCHS = 20
CLASSIFIER_EPOCHS = 80
PRETRAIN_EPOCH = 25
FINAL_CLASSIFIER_EPOCHS = 100
SUBEPOCH = 10
BATCH_SIZE = 250
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096
ALPHA = 0.2

torch.set_num_threads(16)
device='cuda:1'

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
        #nn.init.kaiming_normal_(self.model_0.weight)
        #nn.init.zeros_(self.model_0.bias)
        self.model_1 = nn.Linear(16, data_shape)
        #nn.init.kaiming_normal_(self.model_1.weight)
        #nn.init.zeros_(self.model_1.bias)
        
        self.time_dim = time_shape        

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, 16)
            #nn.init.kaiming_normal_(self.model_alpha_0.weight)
            #nn.init.zeros_(self.model_alpha_0.bias)
            self.model_alpha_1 = nn.Linear(16, data_shape)
            #nn.init.kaiming_normal_(self.model_alpha_1.weight)
            #nn.init.zeros_(self.model_alpha_1.bias)
            #self.model_alpha_2 = nn.Linear(32, data_shape)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X, times):

        thresholds = self.model_1(self.model_0(times))

        if self.leaky:
            alphas = self.model_alpha_1(self.model_alpha_0(times))
        else:
            alphas = 0.0
        X = torch.where(X>thresholds,X-thresholds,alphas*(X-thresholds))
        return X


class TimeReLUCNN(nn.Module):

    def __init__(self, data_shape, time_shape, leaky=False):
        
        super(TimeReLUCNN,self).__init__()
        self.leaky = leaky
        self.model_0 = nn.Linear(time_shape, 16)
        #nn.init.kaiming_normal_(self.model_0.weight)
        #nn.init.zeros_(self.model_0.bias)
        self.model_1 = nn.Linear(16, data_shape)
        #nn.init.kaiming_normal_(self.model_1.weight)
        #nn.init.zeros_(self.model_1.bias)
        
        self.time_dim = time_shape        

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, 16)
            #nn.init.kaiming_normal_(self.model_alpha_0.weight)
            #nn.init.zeros_(self.model_alpha_0.bias)
            self.model_alpha_1 = nn.Linear(16, data_shape)
            #nn.init.kaiming_normal_(self.model_alpha_1.weight)
            #nn.init.zeros_(self.model_alpha_1.bias)
        
        self.leaky = leaky
        self.time_dim = time_shape
    
    def forward(self, X, times):
        # times = X[:,-self.time_dim:]
        orig_shape = X.size()
        # print(orig_shape)
        #X = X.view(orig_shape[0],-1)
        #if len(times.size()) == 3:
        #    times = times.squeeze(2)
        
        thresholds = self.model_1(F.relu(self.model_0(times)))
        if self.leaky:
            alphas = self.model_alpha_1(F.relu(self.model_alpha_0(times)))
        else:
            alphas = 0.0

        thresholds = thresholds[:,:,None,None]
        alphas = alphas[:,:,None,None]
        # print("Thresh",thresholds.shape,X.size())
        X = torch.where(X>thresholds,X-thresholds,alphas*(X-thresholds))
        # print(X.size())
        #print(X.shape)
        #X = X.view(*list(orig_shape))
        return X

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        #self.relu = TimeReLUCNN()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        #print('Shapes')
        #print(x.shape)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        #print(self.downsample)
        #print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_dim=10):
        super(ResNet, self).__init__()
        self.time_shape = 16
        self.in_channels = 16

        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)
        
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, layers[3], 2)
        
        self.avg_pool = nn.AvgPool2d(2)
        self.fc_time = nn.Linear(self.time_shape, 128 * 7 * 7)
        self.fc1 = nn.Linear(2 * 128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.t2v = Time2Vec(1, self.time_shape)
        
        #self.relu_0 = TimeReLUCNN(16 * 28 * 28, self.time_shape, True)
        #self.relu_1 = TimeReLUCNN(32 * 28 * 28, self.time_shape, True)
        #self.relu_2 = TimeReLUCNN(32 * 14 * 14, self.time_shape, True)
        #self.relu_3 = TimeReLUCNN(64 * 7 * 7, self.time_shape, True)

        self.relu_conv1 = TimeReLUCNN(16, self.time_shape, True)
        self.relu_conv2 = TimeReLUCNN(32, self.time_shape, True)
        self.relu_conv3 = TimeReLUCNN(64, self.time_shape, True)
        self.relu_conv4 = TimeReLUCNN(128, self.time_shape, True)
        self.relu_fc1 = TimeReLU(256, self.time_shape, True)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, times=None):
        #times_ = times.unsqueeze(2).repeat(1,28,28)[:, None, :, :]
        #x = torch.cat([x, times_], dim=1)
        times = self.t2v(times)
        times_ = self.fc_time(times)
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.relu_conv1(out, times)
        out = self.dropout(out)
        #print('L1:', out.shape)
        out = self.layer2(out)
        out = self.relu_conv2(out, times)
        out = self.dropout(out)
        #print('L2:', out.shape)
        out = self.layer3(out)
        out = self.relu_conv3(out, times)
        out = self.dropout(out)
        #print('L3:', out.shape)
        out = self.layer4(out)
        out = self.relu_conv4(out, times)
        #print('L4:',out.shape)
        #print('Out_shape:', out.shape)
        #out = self.avg_pool(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, times_], dim=1)
        #print('Out_shape:', out.shape)
        out = self.fc1(out)
        out = self.relu_fc1(out, times)
        out = self.fc2(out)
        return out


class LeNet(nn.Module):

    def __init__(self, output_dim):

        super().__init__()

        self.time_shape = 16
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)

        self.fc_time = nn.Linear(self.time_shape, 16 * 4 * 4)
        self.fc_1 = nn.Linear(2 * 16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

        self._init_all()

        self.relu_conv1 = TimeReLUCNN(6, self.time_shape, True)
        self.relu_conv2 = TimeReLUCNN(16, self.time_shape, True)
        self.relu_fc1 = TimeReLU(120, self.time_shape, True)
        self.relu_fc2 = TimeReLU(84, self.time_shape, True)
        self.t2v = Time2Vec(1, self.time_shape)
        self.dropout = nn.Dropout(0.25)

    def _init_all(self):

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

        nn.init.kaiming_normal_(self.fc_1.weight)
        nn.init.zeros_(self.fc_1.bias)
        nn.init.kaiming_normal_(self.fc_2.weight)
        nn.init.zeros_(self.fc_2.bias)
        nn.init.kaiming_normal_(self.fc_3.weight)
        nn.init.zeros_(self.fc_3.bias)        

    def forward(self, x, times=None):

        #x = [batch size, 1, 28, 28]
        #times_ = times.unsqueeze(2).repeat(1,28,28)[:, None, :, :]
        #x = torch.cat([x, times_], dim=1)
        times = self.t2v(times)

        times_ = F.leaky_relu(self.fc_time(times))

        x = self.conv1(x) # [B, 6, 24, 24]
        x = F.max_pool2d(x, kernel_size = 2) # [B, 6, 12, 12]
        x = self.relu_conv1(x, times)
        #print('shape relu_conv1', x.shape)
        x = self.dropout(x)
        x = self.conv2(x) # [B, 16, 8, 8]
        
        x = F.max_pool2d(x, kernel_size = 2) #[B, 16, 4, 4]
        x = self.relu_conv2(x, times)
        #print('shape relu_conv2', x.shape)
        x = x.view(x.shape[0], -1) #[B, 256]
        
        h = x
        x = torch.cat([x, times_], dim=1)
        
        x = self.fc_1(x)
        
        #x = [batch size, 120]
        
        x = self.relu_fc1(x, times)

        x = self.fc_2(x)
        
        #x = batch size, 84]
        
        x = self.relu_fc2(x, times)

        x = self.fc_3(x)

        #x = [batch size, output dim]
        
        return x

def train_classifier(X, U, Y, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()
    
    Y_pred = F.softmax(classifier(X, U))
    Y_true = Y
    #Y_true = torch.argmax(Y, 1).view(-1,1).float()
    #print(Y_true.shape, Y_pred.shape)
    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))
    #print(pred_loss)
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

def finetune(X, U, Y, delta, classifier, classifier_optimizer):

    classifier_optimizer.zero_grad()

    U_grad = U.clone() - delta
    U_grad.requires_grad_(True)
    Y_pred = classifier(X, U_grad)
    partial_logit_pred_t = []
    for idx in range(Y.shape[1]):
        logit = Y_pred[:,idx].view(-1,1)
        partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
        #print(partial_logit_pred_t)

    partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)
    #print(partial_Y_pred_t.shape)
    Y_pred = Y_pred + delta * partial_Y_pred_t
    Y_pred = F.softmax(Y_pred)
    Y_true = Y
    #Y_true = torch.argmax(Y, 1).view(-1,1).float()

    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))# + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss

def adversarial_finetune(X, U, Y, delta, classifier, classifier_optimizer):
    
    classifier_optimizer.zero_grad()
    
    delta.requires_grad_(True)
    
    for ii in range(5):

        U_grad = U.clone() - delta
        U_grad.requires_grad_(True)
        Y_pred = classifier(X, U_grad)
        Y_true = torch.argmax(Y, 1).view(-1,1).float()

        partial_logit_pred_t = []
        for idx in range(Y.shape[1]):
            logit = Y_pred[:,idx].view(-1,1)
            partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])

        partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)

        #partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
        Y_pred = Y_pred + delta * partial_Y_pred_t
        Y_pred = F.softmax(Y_pred)
        #print(Y_pred.shape)
        #print(Y_pred[0:10])
        loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))

        partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        delta = delta + 0.05*partial_loss_delta
        #print('%d %f' %(ii, delta.clone().detach().numpy()))
    
    delta = delta.clamp(-0.05, 0.05)
    U_grad = U.clone() - delta
    U_grad.requires_grad_(True)
    Y_pred = classifier(X, U_grad)
    partial_logit_pred_t = []
    for idx in range(Y.shape[1]):
        logit = Y_pred[:,idx].view(-1,1)
        partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
        #print(partial_logit_pred_t)

    partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)
    #print(partial_Y_pred_t.shape)
    Y_pred = Y_pred + delta * partial_Y_pred_t
    Y_pred = F.softmax(Y_pred)
    Y_true = Y
    #Y_true = torch.argmax(Y, 1).view(-1,1).float()

    
    pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))# + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
    #pred_loss = torch.mean((Y_pred - Y_true)**2)
    pred_loss.backward()
    
    classifier_optimizer.step()

    return pred_loss


def train(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

    X_source = X_data[source_indices]
    Y_source = Y_data[source_indices]
    U_source = U_data[source_indices]

    X_target = X_data[target_indices]
    Y_target = Y_data[target_indices]
    U_target = U_data[target_indices]

    print(X_source.shape)
    #classifier = PredictionModel(3, 6, 1)
    #classifier = LeNet(10).to(device)
    net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
    }
    classifier = ResNet(**net_args).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-4)

    X_past = np.vstack(X_source)
    U_past = np.hstack(U_source)
    Y_past = np.vstack(Y_source)

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
                                                    torch.tensor(Y_past).float()), BATCH_SIZE, True)
        class_loss = 0
        for batch_X, batch_U, batch_Y in past_dataset:

            batch_U = batch_U.view(-1,1).to(device)
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            l = train_classifier(batch_X, batch_U, batch_Y, classifier, classifier_optimizer)
            #print(l.detach().cpu().numpy())
            class_step += 1
            class_loss += l
        print("Epoch %d Loss %f"%(epoch,class_loss))

    X_past = X_source[0]
    U_past = U_source[0]
    Y_past = Y_source[0]

    
    print('------------------------------------------------------------------------------------------')
    print('FINETUNING CLASSIFIER')
    print('------------------------------------------------------------------------------------------')

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-5)

    #domains = np.random.randint(0, len(X_source), 10)
    domains = np.arange(2, len(X_source))
    ii = 0
    for index in domains:

        print('Finetuning step %d Domain %d' %(ii, index))
        ii+=1
        print('------------------------------------------------------------------------------------------')
        for epoch in range(FINETUNING_EPOCHS):
            cur_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source[index]).float(), torch.tensor(Y_data[index]).float(), 
                torch.tensor(U_source[index]).float()), BATCH_SIZE, True)

            loss = 0
            for batch_X, batch_Y, batch_U in cur_dataset:

                batch_U = batch_U.view(-1,1).to(device)
                #delta = torch.FloatTensor(1,).uniform_(-0.01, 0.01).to(device)
                delta = torch.tensor([0.0]).to(device)
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                loss += finetune(batch_X, batch_U, batch_Y, delta, classifier, classifier_optimizer)
                
            print("Epoch %d Loss %f"%(epoch,loss))
    
    for i in range(len(X_target)):

        target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(),
                                                        torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,False)
        Y_pred = []
        Y_label = []
        for batch_X, batch_U, batch_Y in target_dataset:


            batch_U = batch_U.view(-1,1).to(device)
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            batch_Y_pred = F.softmax(classifier(batch_X, batch_U)).detach().cpu().numpy()

            Y_pred = Y_pred + [batch_Y_pred]
            #Y_label = Y_label + [batch_Y]

        Y_pred = np.vstack(Y_pred)
        #Y_label = np.vstack(Y_label)

        
        Y_pred = np.argmax(Y_pred, 1)
        Y_true = np.argmax(Y_target[i], 1)

        print(Y_pred.shape)
        print(Y_true.shape)

        print(accuracy_score(Y_true, Y_pred))
        print(confusion_matrix(Y_true, Y_pred))
        print(classification_report(Y_true, Y_pred))    
        #print(np.mean((Y_pred - Y_label)**2))
        #print(np.mean(abs(Y_pred - Y_label)))
        #print(np.mean(abs(Y_pred - Y_label)/Y_label))

    for u in range(num_indices):
        visualize_classifier(classifier, u, X_data, Y_data)

if __name__ == "__main__":
    
    #X_data, Y_data, U_data = load_moons(11)
    X_data, Y_data, A_data, U_data = load_Rot_MNIST(False)
    #X_data, Y_data, A_data, U_data = load_housing('raw_sales.csv')
    train(X_data, Y_data, U_data, 6, [0,1,2,3], [4,5])
    '''
    net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
    }
    model = ResNet(**net_args)
    #print(model)
    x = torch.Tensor(5, 1, 28, 28)
    t = torch.Tensor(5, 1)
    print(model(x, t).shape)
    print(model(x, t))
    '''
