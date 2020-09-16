import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from torch.utils.tensorboard import SummaryWriter
import time 

from models import *
from data_loaders import *

#dump all these to a config file
EPOCH = 500
BATCH_SIZE = 100
torch.set_num_threads(8)

def train_classifier_d(X, Y, classifier, classifier_optimizer, verbose=False):

    classifier_optimizer.zero_grad()
    Y_pred = classifier(X)
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

def train_classifier(X, A, Y, classifier, classifier_optimizer, verbose=False):

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

def no_adaptation(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

    X_source = np.vstack(X_data[source_indices])
    Y_source = np.vstack(Y_data[source_indices])
    U_source = np.hstack(U_data[source_indices])

    X_target = np.vstack(X_data[target_indices])
    Y_target = np.vstack(Y_data[target_indices])
    U_target = np.hstack(U_data[target_indices])

    classifier = ClassifyNet(670,[256, 256, 128], 2)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-3)

    writer = SummaryWriter(comment='{}'.format(time.time()))

    all_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source).float(),
                                                    torch.tensor(U_source).float(), torch.tensor(Y_source).float()),BATCH_SIZE,True)

    print('------------------------------------------------------------------------------------------')
    print('TRAINING')
    print('------------------------------------------------------------------------------------------')
    for epoch in range(EPOCH):
        loss = 0
        for batch_X,batch_U,batch_Y in all_data:
            # step += 1
            loss += train_classifier_d(batch_X, batch_Y, classifier, classifier_optimizer, verbose=False)
        if epoch%10 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

    print('------------------------------------------------------------------------------------------')
    print('TESTING')
    print('------------------------------------------------------------------------------------------')
            
    target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target).float(), torch.tensor(U_target).float(), torch.tensor(Y_target).float()),BATCH_SIZE,False)
    Y_pred = []
    for batch_X, batch_U, batch_Y in target_dataset:

        batch_Y_pred = classifier(batch_X).detach().cpu().numpy()

        Y_pred = Y_pred + [batch_Y_pred]  
    Y_pred = np.vstack(Y_pred)
    print('shape: ',Y_pred.shape)
    # print(Y_pred)
    Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
    Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

    # print(Y_pred-Y_true)
    print(accuracy_score(Y_true, Y_pred))
    print(confusion_matrix(Y_true, Y_pred))
    print(classification_report(Y_true, Y_pred))    



def incremental_finetuning(X_data,Y_data,A_data,U_data,num_indices, source_indices, target_indices):

    ## BASELINE 1- Sequential training with no adaptation ##

    X_source = X_data[source_indices]
    Y_source = Y_data[source_indices]
    U_source = U_data[source_indices]
    A_source = A_data[source_indices]

    X_target = X_data[target_indices]
    Y_target = Y_data[target_indices]
    U_target = U_data[target_indices]
    A_target = A_data[target_indices]
   
    classifier = ClassifyNet(671,[256, 256, 128],2, True, True)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-3)

    writer = SummaryWriter(comment='{}'.format(time.time()))

        
    for i in source_indices:

        X_past = X_source[i]
        U_past = U_source[i]
        Y_past = Y_source[i]
        A_past = A_source[i]
        
        past_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(A_past).float(),
                                                        torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)

        print('------------------------------------------------------------------------------------------')
        print('TRAINING - DOMAIN: %d' % i)
        print('------------------------------------------------------------------------------------------')
        for epoch in range(EPOCH):
            loss = 0
            for batch_X,batch_A,batch_U,batch_Y in past_data:
                batch_A = batch_A.view(-1,1)
                batch_X = torch.cat([batch_X, batch_A], dim=1)
                # step += 1
                loss += train_classifier(batch_X, batch_A, batch_Y, classifier, classifier_optimizer, verbose=False)
            if epoch%10 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

    print('------------------------------------------------------------------------------------------')
    print('TESTING')
    print('------------------------------------------------------------------------------------------')
    for i in range(len(X_target)):
        
        target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(A_target[i]).float(),
                                                            torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,False)
        Y_pred = []
        for batch_X, batch_A, batch_U, batch_Y in target_dataset:

            batch_A = batch_A.view(-1,1)
            batch_X = torch.cat([batch_X, batch_A], dim=1)
            batch_Y_pred = classifier(batch_X, batch_A).detach().cpu().numpy()

            Y_pred = Y_pred + [batch_Y_pred]  
        Y_pred = np.vstack(Y_pred)
        print('shape: ',Y_pred.shape)
        # print(Y_pred)
        Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
        Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target[i]])

        # print(Y_pred-Y_true)
        print(accuracy_score(Y_true, Y_pred))
        print(confusion_matrix(Y_true, Y_pred))
        print(classification_report(Y_true, Y_pred))    
    
    

if __name__ == "__main__":

    X_data, Y_data, A_data, U_data = load_sleep2('shhs1-dataset-0.15.0.csv')
    X_data = preprocess_sleep2(X_data, [3])
    #incremental_finetuning(X_data, Y_data, A_data, U_data, 5, [0, 1, 2, 3], [4])
    no_adaptation(X_data, Y_data, U_data, 5, [0,1,2,3], [4])