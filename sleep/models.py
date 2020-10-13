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
        self.model_0 = nn.Linear(time_shape, 32)
        self.model_1 = nn.Linear(32, data_shape)
        #self.model_2 = nn.Linear(32, data_shape)
        self.time_dim = time_shape        

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, 32)
            self.model_alpha_1 = nn.Linear(32, data_shape)
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

class Autoencoder(nn.Module):

    def __init__(self, data_shape, hidden_shape_list, out_shape, enc_index, time_conditioning=False, leaky=False, time2vec=False):
        
        super(Autoencoder,self).__init__()

        # Disallow simple logistic regression
        assert (len(hidden_shape_list) > 0)
        self.time_conditioning = time_conditioning
        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.time2vec = None
        self.using_time2vec = time2vec
        self.hidden_layers = len(hidden_shape_list)
        self.time_dim = 1
        self.enc_index = enc_index
        if time2vec:
            self.time2vec = Time2Vec(1, 8)
            self.time_dim = 8
        
        self.layers.append(nn.Linear(data_shape,hidden_shape_list[0]))
        if time_conditioning:
            self.relus.append(TimeReLU(hidden_shape_list[0], self.time_dim, leaky))
        else:
            self.relus.append(nn.LeakyReLU())

        for i in range(1, len(hidden_shape_list)):
            self.layers.append(nn.Linear(hidden_shape_list[i-1], hidden_shape_list[i]))
            if time_conditioning:
                self.relus.append(TimeReLU(hidden_shape_list[i], self.time_dim, leaky))
            else:
                self.relus.append(nn.LeakyReLU())
            
        self.layers.append(nn.Linear(hidden_shape_list[-1], out_shape))
        self.relus.append(nn.LeakyReLU())

    def _reset_layers(self):

        for l in self.layers[1:]:
            l.reset_parameters()

    def forward(self, X, times=None):
        
        if self.time_conditioning:

            if self.using_time2vec:
                times = self.time2vec(times)

            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X), times)
        else:
            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X))
            
        X = self.relus[-1](self.layers[-1](X))
        X = torch.softmax(X, dim=1)

        return X

    def _latent(self, X, times=None):

        if self.time_conditioning:

            if self.using_time2vec:
                times = self.time2vec(times)

            for i in range(self.enc_index):
                X = self.relus[i](self.layers[i](X), times)
        else:
            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X))

        return X

class ClassifyNet(nn.Module):

    def __init__(self, data_shape, hidden_shape_list, out_shape, time_conditioning=False, leaky=False, time2vec=False):
        
        super(ClassifyNet,self).__init__()

        # Disallow simple logistic regression
        assert (len(hidden_shape_list) > 0)
        self.time_conditioning = time_conditioning
        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.time2vec = None
        self.using_time2vec = time2vec
        self.hidden_layers = len(hidden_shape_list)
        self.time_dim = 1
        if time2vec:
            self.time2vec = Time2Vec(1, 8)
            self.time_dim = 8
        
        self.layers.append(nn.Linear(data_shape,hidden_shape_list[0]))
        if time_conditioning:
            self.relus.append(TimeReLU(hidden_shape_list[0], self.time_dim, leaky))
        else:
            self.relus.append(nn.LeakyReLU())

        for i in range(1, len(hidden_shape_list)):
            self.layers.append(nn.Linear(hidden_shape_list[i-1], hidden_shape_list[i]))
            if time_conditioning:
                self.relus.append(TimeReLU(hidden_shape_list[i], self.time_dim, leaky))
            else:
                self.relus.append(nn.LeakyReLU())
            
        self.layers.append(nn.Linear(hidden_shape_list[-1], out_shape))
        self.relus.append(nn.LeakyReLU())

    def _reset_layers(self):

        for l in self.layers[1:]:
            l.reset_parameters()

    def forward(self, X, times=None):
        
        if self.time_conditioning:

            if self.using_time2vec:
                times = self.time2vec(times)

            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X), times)
        else:
            for i in range(self.hidden_layers):
                X = self.relus[i](self.layers[i](X))
            
        X = self.relus[-1](self.layers[-1](X))
        X = torch.softmax(X, dim=1)

        return X


class Transformer(nn.Module):
    def __init__(self, data_shape, latent_list, time2vec=False, label_dim=0):
        
        super(Transformer,self).__init__()

        self.using_time2vec = time2vec
        self.time2vec = None
        self.time_dim = 2
        self.latent_layers = len(latent_list)
        if self.using_time2vec:
            self.time2vec = Time2Vec(2, 8)
            self.time_dim = 8

        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.layers.append(nn.Linear(data_shape, latent_list[0]))
        self.relus.append(TimeReLU(latent_list[0], self.time_dim, True))
        self.bn.append(nn.BatchNorm1d(latent_list[0], affine=True))

        for i in range(1, self.latent_layers):
            self.layers.append(nn.Linear(latent_list[i-1], latent_list[i]))
            self.relus.append(TimeReLU(latent_list[i], self.time_dim))
            self.bn.append(nn.BatchNorm1d(latent_list[i], affine=True))
        self.layers.append(nn.Linear(latent_list[-1],data_shape-2))

        self.label_dim = label_dim

    def forward(self, X, times):
        
        if self.using_time2vec:
            times = self.time2vec(times)

        for i in range(self.latent_layers):
            X = self.bn[i](self.relus[i](self.layers[i](X), times))

        X = self.layers[-1](X)
        return X


class Discriminator(nn.Module):

    def __init__(self,data_shape, hidden_shape_list, time_conditioning=False, time2vec=False, is_wasserstein=True):

        super(Discriminator,self).__init__()
        
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.relu = nn.LeakyReLU()
        self.relus = nn.ModuleList()
        self.time_conditioning = time_conditioning
        

        self.hidden_layers = len(hidden_shape_list)
        self.using_time2vec = time2vec
        self.time2vec = None
        self.time_dim = 1
        if self.using_time2vec:
            self.time2vec = Time2Vec(1, 8)
            self.time_dim = 8

        self.layers.append(nn.Linear(data_shape,hidden_shape_list[0]))
        self.bn.append(nn.BatchNorm1d(hidden_shape_list[0], affine=True))

        if self.time_conditioning:
            self.relus.append(TimeReLU(hidden_shape_list[0], self.time_dim))

        for i in range(1, self.hidden_layers):
            self.layers.append(nn.Linear(hidden_shape_list[i-1], hidden_shape_list[i]))
            self.bn.append(nn.BatchNorm1d(hidden_shape_list[i], affine=True))
            if self.time_conditioning:
                self.relus.append(TimeReLU(hidden_shape_list[i], self.time_dim))
            
        self.layers.append(nn.Linear(hidden_shape_list[-1], 1))
        
        self.is_wasserstein = is_wasserstein

    def _load_time2vec_model(self, model):

        self.time2vec.load_state_dict(model.state_dict())
        for param in self.time2vec.parameters():
            param.requires_grad = False

    def forward(self,X,times=None):
        
        if self.using_time2vec and self.time_conditioning:
            times = self.time2vec(times)

        for i in range(self.hidden_layers):
            if self.time_conditioning:
                X = self.bn[i](self.relus[i](self.layers[i](X), times))
            else:
                X = self.bn[i](self.relu(self.layers[i](X)))

        X = self.layers[-1](X)
        if not self.is_wasserstein:
            X = torch.sigmoid(X)
            
        return X

    def _feature(self, X, times=None):

        if self.using_time2vec and self.time_conditioning:
            times = self.time2vec(times)

        for i in range(self.hidden_layers):
            if self.time_conditioning:
                X = self.bn[i](self.relus[i](self.layers[i](X), times))
            else:
                X = self.bn[i](self.relu(self.layers[i](X)))

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


#def classification_loss(Y_pred, Y):
    # print(Y_pred)
#    return  -1.*torch.sum((Y * torch.log(Y_pred + 1e-9)))

def classification_loss(Y_pred, Y, is_kernel=False, kernel=None):
    
    loss = - torch.sum((Y * torch.log(Y_pred + 1e-9)), 1).view(-1,1)
    
    if is_kernel:
        loss = torch.sum(loss * kernel)

    loss = torch.sum(loss)
    return loss

def bxe(real, fake):
    return -1.*((real*torch.log(fake+ 1e-9)) + ((1-real)*torch.log(1-fake + 1e-9)))

def discriminator_loss(real_output, trans_output, is_wasserstein):


    if is_wasserstein:

        real_loss = torch.mean(real_output)
        trans_loss = -torch.mean(trans_output)
        total_loss = real_loss + trans_loss

    else:
        real_loss = bxe(torch.ones_like(real_output), real_output)
        trans_loss = bxe(torch.zeros_like(trans_output), trans_output)
        total_loss = real_loss + trans_loss
        total_loss = total_loss.mean()
        
    return total_loss
'''
def discriminator_loss_wasserstein(real_output, trans_output):

    # bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    
    return total_loss

'''
def reconstruction_loss(x,y):
    
    loss = torch.sum((x-y)**2,dim=1)
    return torch.mean(loss)


def transformer_loss(trans_output,is_wasserstein=True):

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

def ot_transformer_loss(real_data, trans_data, disc_output, ot_data, is_wasserstein=True):

    time_diff = torch.exp(-torch.abs(real_data[:,-1] - real_data[:,-2])/2).view(-1,1)
    
    re_loss = torch.sum((trans_data - ot_data)**2, 1).view(-1,1)
    disc_loss = transformer_loss(disc_output, is_wasserstein).view(-1,1)
    loss = re_loss * (1 - time_diff) + disc_loss * time_diff
    
    return torch.mean(loss)


