# Import file functions and classes
from data import GetTarget, QM9DataModule, AtomwisePostProcessing  
from utils import LocalEdges, RadialBasis, CosineCutoff
from blocks import MessageBlock, UpdateBlock
from model import PaiNN
from parser import cli
from trainer import training, test
from plots import simple_loss_plot, true_pred_plot

# Import packages
import torch
import csv
import pandas as pd
from pytorch_lightning import seed_everything
from tqdm import trange
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# This comes from "Training and testing" cell from notebook.py 

def main():
    # set working directory
    os.chdir('PaiNN-Project') # comment out if working directory already is git folder

    args = []
    args = cli(args)
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and prepare data from the QM9 data set.
    dm = QM9DataModule(
        target=args.target,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
        batch_size_inference=args.batch_size_inference,
        num_workers=args.num_workers,
        splits=args.splits,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    dm.prepare_data()
    dm.setup()

    # Calculate target statistics.
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )
    print('length of test set')
    print(len(dm.data_test))
    # Initialize the model.
    painn = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,     # 3
        num_features=args.num_features,                                 # 128
        num_outputs=args.num_outputs,                                   # 1
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,                                   # 5
        device=device
    )

    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    )

    painn.to(device)
    post_processing.to(device)

    # Define optimizer.
    optimizer = torch.optim.AdamW(
        painn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(optimizer = optimizer,
                                  mode = 'min',
                                  factor = 0.1)
    
    # Train the model.
    pbar = trange(args.num_epochs)

    losses_train, losses_eval = training(
        epoch_range=pbar, 
        model=painn, 
        optimizer=optimizer, 
        post_processing=post_processing, 
        dm = dm, 
        device=device,
        scheduler=scheduler
    )

    mae, predictions, true_labels = test(
        model=painn, 
        dm=dm, 
        post_processing=post_processing, 
        device=device
    )

    unit_conversion = dm.unit_conversion[args.target]
    MAE = unit_conversion(mae)

    os.makedirs('output', exist_ok=True) 
    
    with open('output/train_eval_MAE.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(losses_train)  # Write list1
        writer.writerow(losses_eval)   # Write list2

    with open('output/test_MAE.csv', "w", newline="") as file:
        file.write(str(MAE))

    # make a pandas dataframe and save it to a csv
    data1 = {
        'Training loss': losses_train,
        'Validation loss': losses_eval
    }
    df = pd.DataFrame(data1)
    df.to_csv('output/train_val_loss_pandas.csv', index=False)

    # make a pandas dataframe and save it to a csv
    data2 = {
        'True Labels': true_labels,
        'Predictions': predictions
    }
    df = pd.DataFrame(data2)
    df.to_csv('output/predictions_vs_true_labels_pandas.csv', index=False)
    
    with open('output/preds_and_true_labels.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(predictions)
        writer.writerow(true_labels)

    simple_loss_plot(losses_train, losses_eval)
    true_pred_plot(predictions=predictions, true_labels=true_labels)

if __name__ == "__main__":
    main()
