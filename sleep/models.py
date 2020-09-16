# # Contains the model definition for the classifier
# # We are considering two approaches - 
# # 1. Transformer does heavy lifting and just gives a transformed distribution to the classifier
# # 2. Classifier does heavy lifting and tries to learn P(y|X,t) with special emphasis on time = T+1

from torch import nn
import torch

class TimeReLU(nn.Module):

    '''
    A ReLU with threshold and alpha as a function of domain indices.
    '''

    def __init__(self, data_shape, time_shape, leaky=False):
        
        super(TimeReLU,self).__init__()
        self.leaky = leaky
        self.model = nn.Linear(time_shape, data_shape)
        self.time_dim = time_shape        

        if self.leaky:
            self.model_alpha = nn.Linear(time_shape, data_shape)

    def forward(self, X, times):

        thresholds = self.model(times)

        if self.leaky:
            alphas = self.model_alpha(times)
        else:
            alphas = 0.0
        X = torch.where(X>thresholds,X,alphas*X+thresholds)
        return X


class ClassifyNet(nn.Module):

    def __init__(self, data_shape, hidden_shape_list, out_shape, time_conditioning=False, leaky=False):
        
        super(ClassifyNet,self).__init__()

        # Disallow simple logistic regression
        assert (len(hidden_shape_list) > 0)
        self.time_conditioning = time_conditioning
        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.hidden_layers = len(hidden_shape_list)
        
        self.layers.append(nn.Linear(data_shape,hidden_shape_list[0]))
        if time_conditioning:
            self.relus.append(TimeReLU(hidden_shape_list[0], 1, leaky))
        else:
            self.relus.append(nn.LeakyReLU())

        for i in range(1, len(hidden_shape_list)):
            self.layers.append(nn.Linear(hidden_shape_list[i-1], hidden_shape_list[i]))
            if time_conditioning:
                self.relus.append(TimeReLU(hidden_shape_list[i], 1, leaky))
            else:
                self.relus.append(nn.LeakyReLU())
            
        self.layers.append(nn.Linear(hidden_shape_list[-1], out_shape))
        self.relus.append(nn.LeakyReLU())

    def forward(self, X, times=None):
        
        if self.time_conditioning:

            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X), times)
        else:
            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X))
            
        X = self.relus[-1](self.layers[-1](X))
        X = torch.softmax(X, dim=1)

        return X


class Transformer(nn.Module):
    def __init__(self, data_shape, latent_shape_1, latent_shape_2, label_dim=0):
        
        super(Transformer,self).__init__()

        self.layer_0 = nn.Linear(data_shape,latent_shape_1)
        self.leaky_relu_0 = TimeReLU(latent_shape_1,2,True)

        self.layer_1 = nn.Linear(latent_shape_1,latent_shape_1)
        self.leaky_relu_1 = TimeReLU(latent_shape_1,2,True)

        self.layer_2 = nn.Linear(latent_shape_1,latent_shape_1)
        self.leaky_relu_2 = TimeReLU(latent_shape_1,2,True)

        self.layer_3 = nn.Linear(latent_shape_1,latent_shape_2)
        self.leaky_relu_3 = TimeReLU(latent_shape_2,2,True)

        self.layer_4 = nn.Linear(latent_shape_2,latent_shape_2)
        self.leaky_relu_4 = TimeReLU(latent_shape_2,2,True)

        self.layer_5 = nn.Linear(latent_shape_2,latent_shape_2)
        self.leaky_relu_5 = TimeReLU(latent_shape_2,2,True)

        self.layer_6 = nn.Linear(latent_shape_2,latent_shape_2)
        self.leaky_relu_6 = TimeReLU(latent_shape_2,2,True)

        self.layer_last = nn.Linear(latent_shape_2,data_shape-2)
        self.label_dim = label_dim

    def forward(self, X, times):
        
        X = self.leaku_relu_0(self.layer_0(X), times)
        X = self.leaku_relu_1(self.layer_1(X), times)
        X = self.leaku_relu_2(self.layer_2(X), times)
        X = self.leaku_relu_3(self.layer_3(X), times)
        X = self.leaku_relu_4(self.layer_4(X), times)
        X = self.leaku_relu_5(self.layer_5(X), times)
        X = self.leaku_relu_6(self.layer_6(X), times)
        
        X = self.layer_last(X)
        if self.label_dim:
            lab = torch.sigmoid(X[:,-1])
            # X_new = self.leaky_relu(X[:,:-1])
            X = torch.cat([X,lab.unsqueeze(1)],dim=1)
        # else:
        #     X = self.leaky_relu(X)
        return X


class Discriminator(nn.Module):

    def __init__(self,data_shape, hidden_shape_list, is_wasserstein=False):

        super(Discriminator,self).__init__()
        
        self.layers = []
        self.hidden_layers = len(hidden_shape_list)

        self.layers = []
        self.relu = nn.LeakyReLU()
        self.layers.append(nn.Linear(data_shape,hidden_shape_list[0]))
        for i in range(1, self.hidden_layers):
            self.layers.append(nn.Linear(hidden_shape_list[i-1], hidden_shape_list[i]))
        
        self.is_wasserstein = is_wasserstein

    def forward(self,X):
        
        for i in range(self.hidden_layers):
            X = self.relu(self.layers[i](X))

        if not self.is_wasserstein:
            X = torch.sigmoid(X)
            
        return X




class TimeEncodings(nn.Module):
    def __init__(self,model_dim,time_dim):
        super(TimeEncodings,self).__init__()
        self.model_dim = model_dim
        self.time_dim = time_dim
    
    def forward(self,X):
        times = X[:,-time_dim:]
        n,_ = X.size()
        freq_0 = torch.tensor([1/(100**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(n,1)

        offsets = torch.ones_like(X) * (3.1415/2)
        offsets[:,::2] = 0.0
        positional_0 = torch.sin(freq_0 * times[:,:1] + offsets)
        if self.time_dim == 2:
            freq_1 = torch.tensor([1/(50**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(n,1)
            positional_1 = torch.sin(freq_1 * times[:,:1] + offsets)
        else:
            positional_1 = torch.zeros_like(positional_0)
        X = X + positional_0 + positional_1
        return X
# class TimeEmbeddings(nn.Module):
#     def __init__(self,model_dim,encoding):


def classification_loss(Y_pred, Y):
    # print(Y_pred)
    return  -1.*torch.sum((Y * torch.log(Y_pred + 1e-9)))

def bxe(real, fake):
    return -1.*((real*torch.log(fake+ 1e-9)) + ((1-real)*torch.log(1-fake + 1e-9)))

def discriminator_loss(real_output, trans_output):

    # bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = bxe(torch.ones_like(real_output), real_output)
    trans_loss = bxe(torch.zeros_like(trans_output), trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss.mean()
def discriminator_loss_wasserstein(real_output, trans_output):

    # bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = torch.mean(real_output)
    trans_loss = -torch.mean(trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss

def reconstruction_loss(x,y):
    # print(torch.cat([x,y],dim=1))
    return torch.sum((x-y)**2,dim=1)


def transformer_loss(trans_output,is_wasserstein=False):

    if is_wasserstein:
        return trans_output
    return bxe(torch.ones_like(trans_output), trans_output)

def discounted_transformer_loss(real_data, trans_data, ot_data, trans_output, pred_class, actual_class,is_wasserstein=False):

    time_diff = torch.exp(-(real_data[:,-1] - real_data[:,-2]))


    #re_loss = reconstruction_loss(ot_data, trans_data)
    tr_loss = transformer_loss(trans_output,is_wasserstein)
    # transformed_class = trans_data[:,-1].view(-1,1)
    re_loss = tr_loss

    # print(actual_class,pred_class)
    class_loss = bxe(actual_class,pred_class)
    
    #loss = torch.mean(1.*time_diff * tr_loss + (1-time_diff) * re_loss + 0.5*class_loss)
    #loss = torch.mean(1.*time_diff * tr_loss + (1-time_diff) * re_loss + 0.5*class_loss)
    loss = torch.mean(tr_loss + 0.5*class_loss)
    # loss = tr_loss.mean()
    return loss, tr_loss.mean(),re_loss.mean(), class_loss.mean()
    #return loss, tr_loss.mean(),0, class_loss.mean()

def ot_transformer_loss(real_data, trans_data, ot_data):

    time_diff = torch.exp(-torch.abs(real_data[:,-1] - real_data[:,-2]))
    re_loss = (trans_data - ot_data)**2

    loss = re_loss * time_diff

    return torch.mean(loss)


