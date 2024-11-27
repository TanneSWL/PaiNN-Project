'''
This file is a training class :) 

'''
from data import GetTarget, QM9DataModule, AtomwisePostProcessing  
from model import PaiNN

import torch
import torch.nn.functional as F

def training(epoch_range, model, optimizer, post_processing, dm, device, scheduler):
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

    return losses_train, losses_eval

def test(model, dm, post_processing, device):
    
    mae = 0
    
    model.eval()
    
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
    
    mae /= len(dm.data_test)
            
    return mae

    
    
