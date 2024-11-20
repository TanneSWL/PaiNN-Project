'''
This file is a training class :) 

'''
from data import GetTarget, QM9DataModule, AtomwisePostProcessing  
from model import PaiNN

import torch
import torch.functional as F

def training(epoch_range, model, optimizer, post_processing, dm, device):
    '''
    This function performs the training loop
    '''
    model.train() # Voers add
    
    losses = [] #Vores add
    for epoch in epoch_range:

        loss_epoch = 0.
        for batch in dm.train_dataloader():
            batch = batch.to(device)

            atomic_contributions = model(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss_step.detach().item()
        loss_epoch /= len(dm.data_train)
        losses.append(loss_epoch) # Vores add - loss for hver epoch
        epoch_range.set_postfix_str(f'Train loss: {loss_epoch:.3e}')
    return losses

def evaluate(model, dm, post_processing, device):
    mae = 0
    model.eval()
    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = batch.to(device)

            atomic_contributions = model(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            mae += F.l1_loss(preds, batch.y, reduction='sum')
            
    return mae

    mae /= len(dm.data_test)
    unit_conversion = dm.unit_conversion[args.target]
    print(f'Test MAE: {unit_conversion(mae):.3f}')
    