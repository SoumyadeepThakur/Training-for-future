import numpy as np
import scipy as sc
import pandas as pd
import argparse
import math
import os
import matplotlib.pyplot as plt
#from transport import *
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from torch.utils.tensorboard import SummaryWriter
import time 
from ot.da import *
from transport import *
from models_new import *
from data_loaders import *
#from regularized_ot import *

ORDINAL_EPOCHS = 50
CLASSIFIER_EPOCHS = 50
SUBEPOCH = 10
BATCH_SIZE = 100
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096
ALPHA = 0.2

torch.set_num_threads(6)

def train_classifier(X, A, U, Y, classifier, encoder, classifier_optimizer):

    classifier_optimizer.zero_grad()
    X_pred = encoder(X, A)
    age_info = A[:,-1].view(-1,1)
    X_pred_age_info = torch.cat([X_pred, age_info], dim=1)
    Y_pred = classifier(X_pred_age_info, age_info)
    
    pred_loss = classification_loss(Y_pred, Y)/BATCH_SIZE
    pred_loss = pred_loss.sum()
    pred_loss.backward()
    classifier_optimizer.step()
    

    return pred_loss


def train_simple_classifier(X, A, Y, classifier, classifier_optimizer,verbose=False):

    classifier_optimizer.zero_grad()
    Y_pred = classifier(X, A)
    pred_loss = classification_loss(Y_pred, Y)/BATCH_SIZE
    # pred_loss = pred_loss.sum()
    pred_loss.backward()
    
    if verbose:
        # print(torch.cat([Y_pred, Y, Y*torch.log(Y_pred),
        # (Y*torch.log(Y_pred)).sum().unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1) ],dim=1))
        for p in classifier.parameters():
            print(p.data)
            print(p.grad.data)
            print("____")
    classifier_optimizer.step()

    return pred_loss

def visualize(X, A, Y, X1, A1, Y1, encoder, classifier, ord_classifier):

    A = A.view(-1,1)
    A1 = A1.view(-1,1)

    # Make encoder and classifier parameters non trainable

    #for p in encoder.parameters():
    #    p.requires_grad = False


    ###
    # Loss = (t - t')^2

    E = encoder(X, A).detach().clone()
    E1 = encoder(X, A).detach().clone()
    E1_actual = encoder(X1, A1).detach().clone()

    E1.requires_grad = True

    time_diff_pred = ord_classifier(E, E1)
    time_diff_true = A1 - A
    
    time_loss = torch.sum((time_diff_true - time_diff_pred)**2)
        
    grads_time = torch.autograd.grad(time_loss, E1, retain_graph=True)[0].data

    with torch.no_grad():

        E_O = E1 - 60.0 * grads_time
        E_O = E_O.detach().clone()

    E = encoder(X, A).detach().clone()
    E.requires_grad = True

    Y_pred = classifier(E, A)
    class_loss = torch.sum(classification_loss(Y_pred, Y))
    grads_class = torch.autograd.grad(class_loss, E)[0].data
    print(grads_class)
    with torch.no_grad():

        E_C = E + 1e-1 * grads_class
        E_C = E_C.detach().clone()    


    Y = np.array([0 if y[0] > y[1] else 1 for y in Y])
    Y1 = np.array([0 if y[0] > y[1] else 1 for y in Y1])
    YO = Y+2
    YC = Y+2
    plt.scatter(E.detach().numpy()[:,0], E.detach().numpy()[:,1], c=Y)
    plt.show()
    plt.clf()
    plt.scatter(E1_actual.detach().numpy()[:,0], E1_actual.detach().numpy()[:,1], c=Y1)
    plt.show()
    plt.clf()
    plt.scatter(E_O.detach().numpy()[:,0], E_O.detach().numpy()[:,1], c=Y)
    plt.show()
    plt.clf()
    E_app = np.vstack([E1_actual.detach().numpy(), E_O.detach().numpy()])
    Y_app = np.hstack([Y1, YO])
    plt.scatter(E_app[:,0], E_app[:,1], c=Y_app)
    plt.show()
    plt.clf()
    E_app2 = np.vstack([E1_actual.detach().numpy(), E_C.detach().numpy()])
    Y_app2 = np.hstack([Y1, YC])
    plt.scatter(E_app2[:,0], E_app2[:,1], c=Y_app2)
    plt.show()
    plt.clf()
def train_ordinal(X, A, X1, A1, encoder, encoder_optimizer, ord_classifier, ord_optimizer, verbos=False):

    ord_optimizer.zero_grad()
    #encoder_optimizer.zero_grad()

    E = encoder(X, A)
    E1 = encoder(X1, A1)

    time_diff_pred = ord_classifier(E, E1)
    time_diff_true = A1 - A

    time_loss = torch.sum((time_diff_true - time_diff_pred)**2)

    time_loss.backward()

    ord_optimizer.step()
    #encoder_optimizer.step()

    return time_loss

def train_crossgrad(X, A, Y, X1, A1, encoder, encoder_optimizer, classifier, classifier_optimizer, ord_classifier, ord_optimizer, verbose=False):

    classifier_optimizer.zero_grad()
    ord_optimizer.zero_grad()
    #encoder_optimizer.zero_grad()
    # Make encoder and classifier parameters non trainable

    #for p in encoder.parameters():
    #    p.requires_grad = False


    ###
    # Loss = (t - t')^2

    E = encoder(X, A).detach().clone()
    E1 = encoder(X, A).detach().clone()
    E1_actual = encoder(X1, A1).detach().clone()

    E1.requires_grad = True

    time_diff_pred = ord_classifier(E, E1)
    time_diff_true = A1 - A
    time_loss = torch.sum((time_diff_true - time_diff_pred)**2)
        
    grads_time = torch.autograd.grad(time_loss, E1, retain_graph=True)[0].data
    #print(E1)
    #print(grads_time)
    with torch.no_grad():

        E_O = E1 - 7.5 * grads_time
        E_O = E_O.detach().clone()

    E = encoder(X, A).detach().clone()
    E.requires_grad = True

    Y_pred = classifier(E, A)
    class_loss = torch.sum(classification_loss(Y_pred, Y))
    grads_class = torch.autograd.grad(class_loss, E)[0].data
    
    with torch.no_grad():

        E_C = E + 1.0 * grads_class
        E_C = E_C.detach().clone()    

    classifier_optimizer.zero_grad()
    ord_optimizer.zero_grad()

    Y_pred = classifier(E, A)
    Y_pred_O = classifier(E_O, A1)
    time_diff_shifted = ord_classifier(E, E_C)

    pred_loss = torch.sum((1-ALPHA) * classification_loss(Y_pred, Y) + ALPHA * classification_loss(Y_pred_O, Y))
    ord_loss = torch.sum((1-ALPHA) * (time_diff_pred - time_diff_true)**2 + ALPHA * (time_diff_shifted - time_diff_true)**2)
    
    loss = pred_loss + ord_loss
    loss.backward()

    classifier_optimizer.step()
    #encoder_optimizer.step()
    ord_optimizer.step()

    return pred_loss, ord_loss

def encoder(X, times=None):
    return X[:,:-1]


def train(X_data, Y_data, A_data, U_data, num_indices, source_indices, target_indices):

    I_d = np.eye(num_indices)
    rng = 2
    X_source = X_data[source_indices]
    Y_source = Y_data[source_indices]
    U_source = U_data[source_indices]
    A_source = A_data[source_indices]

    X_target = X_data[target_indices]
    Y_target = Y_data[target_indices]
    U_target = U_data[target_indices]
    A_target = A_data[target_indices]

    #encoder = Encoder(671, [384, 384], 256, True, True)
    #classifier = ClassifyNet(256, [128, 128], 2, True, True)
    #ord_classifier = OrdinalClassifier(256, 128, 1, 1)

    #encoder = Encoder(3, [2], 2, True, True)
    classifier = ClassifyNet(2, [6, 4], 2, True, True)
    ord_classifier = OrdinalClassifier(2, 2, 1, 1)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),1e-2)
    #encoder_optimizer = torch.optim.Adam(encoder.parameters(), 5e-4)
    encoder_optimizer = None
    ord_optimizer = torch.optim.Adam(ord_classifier.parameters(), 1e-3)
    #discriminator_optimizer = torch.optim.Adam(classifier.parameters(),1e-4)
    #final_classifier_optimizer = torch.optim.Adam(final_classifier.parameters(), 5e-4)
    

    #X_past = np.vstack(X_source)
    #A_past = np.hstack(A_source)
    #U_past = np.hstack(U_source)
    #Y_past = np.vstack(Y_source)
    X_past = X_source[0]
    A_past = A_source[0]
    U_past = U_source[0]
    Y_past = Y_source[0]
    writer = SummaryWriter(comment='{}'.format(time.time()))

    #ot_maps = [[None for x in range(len(source_indices))] for y in range(len(source_indices))]

    '''
    for i in range(len(source_indices)):
        for j in range(i, len(source_indices)):
            if i!=j:
                Ys = np.array([0 if y[0] > y[1] else 1 for y in Y_data[source_indices[i]]])
                Yt = np.array([0 if y[0] > y[1] else 1 for y in Y_data[source_indices[j]]])
                ot_sinkhorn = RegularizedSinkhornTransportOTDA(reg_e=5.0, max_iter=50, norm='mean', verbose=True)
                ot_sinkhorn.fit(Xs=X_data[source_indices[i]]+1e-6, ys=Ys, Xt=X_data[source_indices[j]]+1e-6, yt=Yt, Xs_trans=X_data[source_indices[i]]+1e-6, 
                                ys_trans=Ys, iteration=0)
                ot_maps[i][j] = ot_sinkhorn.transform(X_data[source_indices[i]])
            else:
                ot_maps[i][j] = X_data[source_indices[i]]
    '''
    # print(ot_maps)
    # assert False
    # print(ot_maps)

    print('------------------------------------------------------------------------------------------')
    print('TRAINING ORDINAL CLASSIFIER')
    print('------------------------------------------------------------------------------------------')

    for index in range(1, len(X_source)):
        class_step = 0
        for epoch in range(0):
            past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(A_past).float(),
                                                            torch.tensor(U_past).float()),BATCH_SIZE,True,drop_last=True)

            X_future = np.vstack(X_source[index:index+rng])
            A_future = np.hstack(A_source[index:index+rng])
            U_future = np.hstack(U_source[index:index+rng])
            Y_future = np.vstack(Y_source[index:index+rng])

            future_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_future).float(), torch.tensor(A_future).float(),
                                                            torch.tensor(U_future).float()),BATCH_SIZE,True,drop_last=True)

            class_loss = 0
            for (batch_X, batch_A, batch_U), (future_X, future_A, future_U) in zip(past_dataset, future_dataset):

                batch_A = batch_A.view(-1,1)
                batch_X = torch.cat([batch_X,batch_A],dim=1)

                future_A = future_A.view(-1,1)
                future_X = torch.cat([future_X,future_A],dim=1)
                #print(batch_U.shape)
                
                ploss = train_ordinal(batch_X, batch_A, future_X, future_A, encoder, encoder_optimizer, ord_classifier, ord_optimizer)
                
            print("Epoch %d Loss %f"%(epoch,ploss))

            X_past = np.vstack([X_past, X_source[index]])
            Y_past = np.vstack([Y_past, Y_source[index]])
            A_past = np.hstack([A_past, A_source[index]])
            U_past = np.hstack([U_past, U_source[index]])


    X_past = X_source[0]
    A_past = A_source[0]
    U_past = U_source[0]
    Y_past = Y_source[0]


    visualize(torch.cat([torch.tensor(X_source[0]).float(), torch.tensor(A_source[0]).float().view(-1,1)], 1), torch.tensor(A_source[0]).float(), torch.tensor(Y_source[0]).float(), 
                            torch.cat([torch.tensor(X_source[2]).float(), torch.tensor(A_source[2]).float().view(-1,1)],1), torch.tensor(A_source[2]).float(), torch.tensor(Y_source[2]).float(), 
                            encoder, classifier, ord_classifier)

    #encoder_optimizer = torch.optim.Adam(encoder.parameters(), 1e-4)
    print('------------------------------------------------------------------------------------------')
    print('TRAINING DOMAIN CLASSIFIERS')
    print('------------------------------------------------------------------------------------------')
    for index in range(1, len(X_source)):
        class_step = 0
        for epoch in range(CLASSIFIER_EPOCHS):
            past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(A_past).float(),
            												torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True,drop_last=True)

            X_future = np.vstack(X_source[index:index+rng])
            A_future = np.hstack(A_source[index:index+rng])
            U_future = np.hstack(U_source[index:index+rng])
            Y_future = np.vstack(Y_source[index:index+rng])

            future_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_future).float(), torch.tensor(A_future).float(),
                                                            torch.tensor(U_future).float(), torch.tensor(Y_future).float()),BATCH_SIZE,True,drop_last=True)

            class_loss = 0
            for (batch_X, batch_A, batch_U, batch_Y), (future_X, future_A, future_U, future_Y) in zip(past_dataset, future_dataset):

                batch_A = batch_A.view(-1,1)
                batch_X = torch.cat([batch_X,batch_A],dim=1)

                future_A = future_A.view(-1,1)
                future_X = torch.cat([future_X,future_A],dim=1)
                #print(batch_U.shape)
                for k in range(10):
                    pl, ol = train_crossgrad(batch_X, batch_A, batch_Y, future_X, future_A, encoder, encoder_optimizer, classifier, classifier_optimizer, ord_classifier, ord_optimizer)
                
            print("Epoch %d Loss %f, %f"%(epoch,pl, ol))

            X_past = np.vstack([X_past, X_source[index]])
            Y_past = np.vstack([Y_past, Y_source[index]])
            A_past = np.hstack([A_past, A_source[index]])
            U_past = np.hstack([U_past, U_source[index]])


    visualize(torch.cat([torch.tensor(X_source[0]).float(), torch.tensor(A_source[0]).float().view(-1,1)], 1), torch.tensor(A_source[0]).float(), torch.tensor(Y_source[0]).float(), 
                            torch.cat([torch.tensor(X_source[2]).float(), torch.tensor(A_source[2]).float().view(-1,1)],1), torch.tensor(A_source[2]).float(), torch.tensor(Y_source[2]).float(), 
                            encoder, classifier, ord_classifier)
    
    #classifier._reset_layers()

    #classifier_optimizer = torch.optim.Adam(classifier.parameters(),1e-2)

    for i in range(len(X_target)):
        print(X_target[i].shape)
        print(Y_target[i].shape)
        print(U_target[i].shape)
        print(A_target[i].shape)
            # print(U_target[i])
        target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(A_target[i]).float(),
                                                        torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,False)
        source_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(A_past).float(),
                                                        torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,False)

        '''
        step = 0
        for epoch in range(FINAL_CLASSIFIER_EPOCHS):

            loss = 0
            
            for batch_X, batch_A, batch_U, batch_Y in source_dataset:
                batch_U = batch_U.view(-1,1)
                batch_A = batch_A.view(-1,1)
                this_U = torch.tensor([U_target[i][0]]*batch_U.shape[0]).float()
                #this_A = torch.tensor([np.mean(A_target[i])]*batch_A.shape[0]).float()
                this_A = torch.tensor(np.random.normal(loc=np.mean(A_source[index]), scale=np.std(A_source[index]), size=batch_A.shape[0])).float()
                this_U = this_U.view(-1,1)
                this_A = this_A.view(-1,1)
                cat_U = torch.cat([batch_U, this_U], dim=1)
                cat_A = torch.cat([batch_A, this_A], dim=1)
                batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
                step += 1
                loss += train_classifier(batch_X, cat_A, cat_U, batch_Y, classifier, encoder, classifier_optimizer)

            # target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,True)
            # source_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
               # print("%f" % loss)
            print('Epoch: %d - Classification Loss: %f' % (epoch, loss))
        '''
        #print(classifier.trainable_variables)
        Y_pred = []
        Y_label = []
        for batch_X, batch_A, batch_U, batch_Y in target_dataset:


            batch_U = batch_U.view(-1,1)
            batch_A = batch_A.view(-1,1)
            batch_X = torch.cat([batch_X, batch_A.view(-1,1)], dim=1)
            batch_Y_pred = classifier(encoder(batch_X, batch_A), batch_A).detach().cpu().numpy()

            Y_pred = Y_pred + [batch_Y_pred]
            Y_label = Y_label + [batch_Y]

        Y_pred = np.vstack(Y_pred)
        Y_label = np.vstack(Y_label)
        print('shape: ',Y_pred.shape)
        # print(Y_pred)
        Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
        Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_label])
        print(accuracy_score(Y_true, Y_pred))
        print(confusion_matrix(Y_true, Y_pred))
        print(classification_report(Y_true, Y_pred))    
    
      
if __name__ == "__main__":

    #X_data, Y_data, A_data, U_data = load_sleep2('shhs1-dataset-0.15.0.csv')
    X_data, Y_data, U_data = load_moons(11)
    #X_data = preprocess_sleep2(X_data, [0, 1])
    
    #train(X_data, Y_data, U_data, U_data, 5, [0, 1], [2])
    train(X_data, Y_data, U_data, U_data, 11, [0,1,2,3,4,5], [6,7])
