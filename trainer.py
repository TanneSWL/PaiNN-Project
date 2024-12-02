'''
This file is a training class :) 

'''
from data import GetTarget, QM9DataModule, AtomwisePostProcessing  
from model import PaiNN

import torch
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        """
        Patience: How many epochs to wait after the last improvement.
        min_delta: Minimum change considered as improvement.
        """
        
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        """
        Checks whether early stopping criteria are met.
        val_loss: Current validation loss.
        epoch: Current epoch.
        """
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch


def training(epoch_range, model, optimizer, post_processing, dm, device, scheduler, early_stopping):
    '''
    This function performs the training loop
    '''
    
    # Create lists to collect the train and evalutation loss for each epoch.
    losses_train = [] 
    losses_eval  = []

    for epoch in epoch_range:

        loss_train_epoch = 0
        loss_eval_epoch  = 0

        #-----------------------------------------------------------------------#
        # TRAIN

        model.train() 
        
        for batch in dm.train_dataloader():
            
            batch = batch.to(device)

            atomic_contributions = model(
                atoms = batch.z,
                atom_positions = batch.pos,
                graph_indexes = batch.batch
            )
            preds = post_processing(
                atoms = batch.z,
                graph_indexes = batch.batch,
                atomic_contributions = atomic_contributions,
            )
            
            #loss_step = F.mse_loss(preds, batch.y, reduction='sum')
            loss_step = F.l1_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_epoch += loss_step.detach().item()
        
        loss_train_epoch /= len(dm.data_train)
        losses_train.append(loss_train_epoch) 
        epoch_range.set_postfix_str(f'Train loss: {loss_train_epoch:.3e}')

        #-----------------------------------------------------------------------#
        # VALIDATION

        model.eval()

        with torch.no_grad():

            for batch in dm.val_dataloader():
                
                batch = batch.to(device)

                atomic_contributions = model(
                    atoms = batch.z,
                    atom_positions = batch.pos,
                    graph_indexes = batch.batch,
                )
                preds = post_processing(
                    atoms = batch.z,
                    graph_indexes = batch.batch,
                    atomic_contributions = atomic_contributions,
                )
                
                loss_step = F.l1_loss(preds, batch.y, reduction='sum')

                loss_eval_epoch += loss_step.detach().item()
    
        loss_eval_epoch /= len(dm.data_val)
        losses_eval.append(loss_eval_epoch) 
        epoch_range.set_postfix_str(f'Evaluation loss: {loss_eval_epoch:.3e}')

        scheduler.step(loss_eval_epoch)

        #-----------------------------------------------------------------------#
        # EARLY STOPPING

        early_stopping(loss_eval_epoch, epoch)
        if early_stopping.early_stop:
            break

    return losses_train, losses_eval

def test(model, dm, post_processing, device):
    
    mae = 0
    
    model.eval()
    predictions = []
    true_labels = []
    smiles_list = []

    with torch.no_grad():
        
        for batch in dm.test_dataloader():
            
            batch = batch.to(device)

            atomic_contributions = model(
                atoms = batch.z,
                atom_positions = batch.pos,
                graph_indexes = batch.batch,
            )
            preds = post_processing(
                atoms = batch.z,
                graph_indexes = batch.batch,
                atomic_contributions = atomic_contributions,
            )
            
            mae += F.l1_loss(preds, batch.y, reduction='sum').detach().item()

            true_label = batch.y.squeeze(1).tolist()
            true_labels.extend(true_label)
            
            smiles = batch.smiles
            smiles_list.extend(smiles)

            prediction = preds.squeeze(1).tolist()
            predictions.extend(prediction)
    
    mae /= len(dm.data_test)
            
    return mae, predictions, true_labels, smiles_list


    
    
