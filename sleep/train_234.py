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
from models import *
from data_loaders import *
#from regularized_ot import *

TRANSFORMER_EPOCH = 3000
CLASSIFIER_EPOCHS = 50
DISCRIMINATOR_EPOCHs = 20
FINAL_CLASSIFIER_EPOCHS = 50
SUBEPOCH = 10
BATCH_SIZE = 100
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096
IS_WASSERSTEIN = True

torch.set_num_threads(4)
def train_transformer_batch(X,A,U,Y,X_transport,real_X,transformer,discriminator,classifier,transformer_optimizer,is_wasserstein=True):

    transformer_optimizer.zero_grad()
    X_pred = transformer(X, U)
    domain_info = U[:,-1].view(-1,1)
    X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)
    pred_disc = discriminator(X_pred_domain_info, domain_info)

    feature_pred = discriminator._feature(X_pred_domain_info, domain_info)
    feature_real = discriminator._feature(real_X, domain_info)
    feature_loss = reconstruction_loss(feature_pred, feature_real)
    ot_loss = ot_transformer_loss(X[:,0:-2], X_pred, pred_disc, X_transport)

    age_info = A[:,-1].view(-1,1)
    X_pred_class = torch.cat([X_pred,age_info],dim=1)
    Y_pred = classifier(X_pred_class, age_info)
    class_loss = classification_loss(Y_pred, Y)
    trans_loss = ot_loss + class_loss + feature_loss
    trans_loss.backward()

    transformer_optimizer.step()

    return trans_loss


def train_discriminator_batch_wasserstein(X_old, A_old, U_old, X_now, transformer, discriminator, discriminator_optimizer):
    
    discriminator_optimizer.zero_grad()

    X_pred_old = transformer(X_old, U_old)
    domain_info = X_old[:,-1].view(-1,1)
    X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

    is_real_old = discriminator(X_pred_old_domain_info, domain_info)
    is_real_now = discriminator(X_now[:,0:-1], domain_info)
    
    disc_loss = discriminator_loss(is_real_now, is_real_old, True)

    disc_loss.backward()

    discriminator_optimizer.step()
    for p in discriminator.parameters():
        p.data.clamp_(-0.5, 0.5)
    return disc_loss

def train_discriminator_batch(X_old, X_now, transformer, discriminator, discriminator_optimizer):

    discriminator_optimizer.zero_grad()
    X_pred_old = transformer(X_old)
    domain_info = X_old[:,-1].view(-1,1)
    X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

    is_real_old = discriminator(X_pred_old_domain_info)
    is_real_now = discriminator(X_now[:,0:-1])
    
    disc_loss = discriminator_loss(is_real_now, is_real_old)

    disc_loss.backward()
    discriminator_optimizer.step()

    return disc_loss


def train_classifier(X, A, U, Y, classifier, transformer, classifier_optimizer):

    classifier_optimizer.zero_grad()
    X_pred = transformer(X, U)
    age_info = A[:,-1].view(-1,1)
    X_pred_age_info = torch.cat([X_pred, age_info], dim=1)
    Y_pred = classifier(X_pred_age_info, age_info)
    
    pred_loss = classification_loss(Y_pred, Y)/BATCH_SIZE

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



def train(X_data, Y_data, A_data, U_data, num_indices, source_indices, target_indices):

    I_d = np.eye(num_indices)

    X_source = X_data[source_indices]
    Y_source = Y_data[source_indices]
    U_source = U_data[source_indices]
    A_source = A_data[source_indices]

    X_target = X_data[target_indices]
    Y_target = Y_data[target_indices]
    U_target = U_data[target_indices]
    A_target = A_data[target_indices]

    transformer = Transformer(672, [384, 256, 128, 256, 384], True)
  
    classifier = ClassifyNet(671, [256, 128, 128], 2, True, True, True)
    final_classifier = ClassifyNet(671, [256, 128, 128], 2)
    discriminator = Discriminator(671, [384, 384, 256, 128], True, True, True)
    transformer_optimizer   = torch.optim.Adam(transformer.parameters(),5e-4)
    classifier_optimizer    = torch.optim.Adam(classifier.parameters(),5e-4)
    discriminator_optimizer = torch.optim.Adam(classifier.parameters(),5e-4)
    final_classifier_optimizer = torch.optim.Adam(final_classifier.parameters(), 5e-4)
    

    X_past = np.vstack(X_source)
    A_past = np.hstack(A_source)
    U_past = np.hstack(U_source)
    Y_past = np.vstack(Y_source)
    writer = SummaryWriter(comment='{}'.format(time.time()))

    ot_maps = [[None for x in range(len(source_indices))] for y in range(len(source_indices))]

    
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
    
    # print(ot_maps)
    # assert False
    # print(ot_maps)
    print('------------------------------------------------------------------------------------------')
    print('TRAINING DOMAIN CLASSIFIERS')
    print('------------------------------------------------------------------------------------------')
    
    class_step = 0
    for epoch in range(CLASSIFIER_EPOCHS):
        past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(A_past).float(),
        												torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
        class_loss = 0
        for batch_X, batch_A, batch_U, batch_Y in past_dataset:

            batch_A = batch_A.view(-1,1)
            batch_X = torch.cat([batch_X,batch_A],dim=1)
            #print(batch_U.shape)
            l = train_simple_classifier(batch_X, batch_A, batch_Y, classifier, classifier_optimizer,verbose=False)
            class_step += 1
            class_loss += l
        print("Epoch %d Loss %f"%(epoch,class_loss))



    X_past = X_source[0]
    W_past = ot_maps[0][0]
    A_past = A_source[0]
    U_past = U_source[0]
    Y_past = Y_source[0]

    discriminator._load_time2vec_model(classifier.time2vec)

    print('------------------------------------------------------------------------------------------')
    print('TRANSFORMER - DISCRIMINATOR LOOP')
    print('------------------------------------------------------------------------------------------')


    for index in range(1, len(X_source)):

        print('Domain %d' %index)
        print('----------------------------------------------------------------------------------------------')

        past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(W_past).float(), torch.tensor(A_past).float(),
                                                    torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
        # present_dataset = torch.utils.data.Dataloader(torch.utils.data.TensorDataset(X_source[index], U_source[index], 
        #                   Y_source[index]),BATCH_SIZE,True,repeat(
        #                   math.ceil(X_past.shape[0]/X_source[index].shape[0])))

        num_past_batches = len(X_past) // BATCH_SIZE
        X_past = np.vstack([X_past, X_source[index]])
        Y_past = np.vstack([Y_past, Y_source[index]])
        A_past = np.hstack([A_past, A_source[index]])
        U_past = np.hstack([U_past, U_source[index]])
        W_past = ot_maps[0][index]
        for j in range(1, index+1):
        	W_past = np.vstack([W_past, ot_maps[j][index]])
        #p = TransformerDataset(X=X_past,
                    #Y=Y_past,U=U_past,source_indices=source_indices,
                    #target_indices=target_indices,this_U=U_source[index][0],index_fun=lambda x: (x %200))
        #print(len(p))
        #all_data = torch.utils.data.DataLoader(p,
         #           BATCH_SIZE,True)            # for batch_X, batch_U, batch_Y, batch_transported in all_dataset:
        print(X_past.shape)
        print(Y_past.shape)
        print(A_past.shape)
        print(U_past.shape)
        print(W_past.shape)
        all_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(W_past).float(), torch.tensor(A_past).float(),
                                                    torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
        num_all_batches  = len(X_past) // BATCH_SIZE
        all_steps_t = 0
        all_steps_d = 0
        step_c = 0

    
        all_dataset_iterator = iter(all_dataset)
        past_dataset_iterator = iter(past_dataset)

        for epoch in range(TRANSFORMER_EPOCH):
            loss_trans, loss_disc = 0,0
            
            loss1, loss2 = 0,0
            step_t,step_d = 0,0

            loop1 = True
            loop2 = True
            
            
            try:
                for j in range(5):
                    batch_X, batch_W, batch_A, batch_U, batch_Y = next(past_dataset_iterator)
                    batch_U = batch_U.view(-1,1)
                    batch_A = batch_A.view(-1,1)
                    this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0])
                    this_A = torch.tensor([np.mean(A_source[index])]*batch_A.shape[0])
                    this_U = this_U.view(-1,1).float()
                    this_A = this_A.view(-1,1).float()
                    cat_U = torch.cat([batch_U, this_U], dim=1)
                    cat_A = torch.cat([batch_A, this_A], dim=1)
                    batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
                    # Do this in a better way

                    indices = np.random.random_integers(0, X_source[index].shape[0]-1, batch_X.shape[0])

                    # Better to shift this to the dataloader
                    real_X = np.hstack([X_source[index][indices], U_source[index][indices].reshape(-1,1), 
                                U_source[index][indices].reshape(-1,1)])
                    real_X = torch.tensor(real_X).float()
                    #if IS_WASSERSTEIN:
                    #    loss_d = train_discriminator_batch_wasserstein(batch_X, real_X, transformer, discriminator, discriminator_optimizer) #train_discriminator_batch(batch_X, real_X)
                    #else:
                    #    loss_d = train_discriminator_batch(batch_X, real_X, transformer, discriminator, discriminator_optimizer)
                    loss_d = train_discriminator_batch_wasserstein(batch_X, cat_A, cat_U, real_X, transformer, discriminator, discriminator_optimizer)
                    loss_disc += loss_d
                    #writer.add_scalar('Loss/disc',loss_d.detach().numpy(),step_d+all_steps_d)
                    #step_d += 1
                    #loop2 = True
                    #else:
                    #loop2 = False
                    #if step_t < num_all_batches:

                
                for j in range(5):
            
                    batch_X, batch_W, batch_A, batch_U, batch_Y = next(all_dataset_iterator)
                
                    batch_U = batch_U.view(-1,1)
                    batch_A = batch_A.view(-1,1)
                    this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0]).float()
                    #this_A = torch.tensor([np.mean(A_source[index])]*batch_A.shape[0]).float()
                    indices = np.random.random_integers(0, X_source[index].shape[0]-1, batch_X.shape[0])

                    # Better to shift this to the dataloader
                    real_X = np.hstack([X_source[index][indices], U_source[index][indices].reshape(-1,1)])
                    real_X = torch.tensor(real_X).float()

                    this_A = torch.tensor(np.random.normal(loc=np.mean(A_source[index]), scale=np.std(A_source[index]), size=batch_A.shape[0])).float()
                    this_U = this_U.view(-1,1)
                    this_A = this_A.view(-1,1)
                    cat_U = torch.cat([batch_U, this_U], dim=1)
                    cat_A = torch.cat([batch_A, this_A], dim=1)
                    # print(batch_X.size(),batch_U.size(),this_U.size(),batch_transported.size())
                    batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
                    loss_t = train_transformer_batch(batch_X,cat_A,cat_U,batch_Y,batch_W,real_X,transformer,discriminator,classifier,transformer_optimizer, is_wasserstein=True)
                    loss_trans += loss_t
                    writer.add_scalar('Loss/transformer',loss_t.detach().numpy(),epoch)

                print('Epoch %d - %9.9f %9.9f' % (epoch, loss_disc.detach().cpu().numpy(), loss_trans.detach().cpu().numpy()))

            except StopIteration:
                all_dataset_iterator = iter(all_dataset)
                past_dataset_iterator = iter(past_dataset)
                epoch -= 1

    
    print('------------------------------------------------------------------------------------------')
    print('TRAINING FINAL CLASSIFIER')
    print('------------------------------------------------------------------------------------------')
    '''
    t1 = torch.tensor(X_source[0]).float()
    print(t1.shape)
    t2 = torch.tensor(U_source[0]).float().view(-1,1)
    print(t2.shape)
    t3 = torch.tensor(U_target[0][0:U_source[0].shape[0]]).float().view(-1,1)
    print(t3.shape)
    
    a = torch.cat([t1, t2, t3], dim=1)
    b = torch.cat([t2, t3], dim=1)
    X_0_t = transformer(a, b).detach().numpy()
    print(X_0_t.shape)
    Y  = np.array([0 if y[0] > y[1] else 1 for y in Y_source[0]])
    plt.scatter(X_source[0][:,0], X_source[0][:,1], c=Y)
    plt.show()
    plt.clf()
    Y  = np.array([0 if y[0] > y[1] else 1 for y in Y_source[0]])
    plt.scatter(X_0_t[:,0], X_0_t[:,1], c=Y)
    plt.show()
    plt.clf()
    Y  = np.array([0 if y[0] > y[1] else 1 for y in Y_target[0]])
    plt.scatter(X_target[0][:,0], X_target[0][:,1], c=Y)
    plt.show()
    '''
    #classifier._reset_layers()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(),1e-2)

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
                loss += train_classifier(batch_X, cat_A, cat_U, batch_Y, classifier, transformer, classifier_optimizer)

            # target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,True)
            # source_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
               # print("%f" % loss)
            print('Epoch: %d - Classification Loss: %f' % (epoch, loss))

        #print(classifier.trainable_variables)
        Y_pred = []
        Y_label = []
        for batch_X, batch_A, batch_U, batch_Y in target_dataset:


            batch_U = batch_U.view(-1,1)
            batch_A = batch_A.view(-1,1)
            batch_X = torch.cat([batch_X, batch_A.view(-1,1)], dim=1)
            batch_Y_pred = classifier(batch_X, batch_A).detach().cpu().numpy()

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

    X_data, Y_data, A_data, U_data = load_sleep2('shhs1-dataset-0.15.0.csv')
    X_data = preprocess_sleep2(X_data, [0, 1])
    
    train(X_data, Y_data, U_data, U_data, 5, [0, 1], [2])
