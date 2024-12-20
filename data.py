'''
This file contains classes related to import and preparation of the data and preprocessing of model output.
'''

# Import packages
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import BaseTransform
from typing import Optional, List, Union, Tuple


class GetTarget(BaseTransform):

    def __init__(self, target: Optional[int] = None) -> None:
        self.target = [target]


    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):

    target_types = ['atomwise' for _ in range(19)]
    target_types[0] = 'dipole_moment'
    target_types[5] = 'electronic_spatial_extent'

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000*t) if i not in [0, 1, 5, 11, 16, 17, 18]
        else (lambda t: t)
        for i in range(19)
    }

    def __init__(
        self,
        target: int = 7,
        data_dir: str = 'data/',
        batch_size_train: int = 100,
        batch_size_inference: int = 1000,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [110000, 10000, 10831],
        seed: int = 0,
        subset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)


    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[:split_idx[0]]
        self.data_val = dataset[split_idx[0]:split_idx[1]]
        self.data_test = dataset[split_idx[1]:]


    def get_target_stats(
        self,
        remove_atom_refs: bool = True,
        divide_by_atoms: bool = True
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        atom_refs = self.data_train.atomref(self.target)                        # Atom reference energy

        ys = list()
        for batch in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(
                    dim=0, index=batch.batch, source=-atom_refs[batch.z]
                )
            if divide_by_atoms:                                                 # Normalize internal energy by the number of atoms
                _, num_atoms  = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs


    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    

class AtomwisePostProcessing(nn.Module):
    """
    Post-processing for (QM9) properties that are predicted as sums of atomic
    contributions.
    """
    
    def __init__(
        self,
        num_outputs: int,
        mean: torch.FloatTensor,
        std: torch.FloatTensor,
        atom_refs: torch.FloatTensor,
    ) -> None:
        """
        Args:
            num_outputs: Integer with the number of model outputs. In most
                cases 1.
            mean: torch.FloatTensor with mean value to shift atomwise
                contributions by.
            std: torch.FloatTensor with standard deviation to scale atomwise
                contributions by.
            atom_refs: torch.FloatTensor of size [num_atom_types, 1] with
                atomic reference values.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.register_buffer('scale', std)
        self.register_buffer('shift', mean)
        self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)


    def forward(
        self,
        atomic_contributions: torch.FloatTensor,
        atoms: torch.LongTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Atomwise post-processing operations and atomic sum.

        Args:
            atomic_contributions: torch.FloatTensor of size [num_nodes,
                num_outputs] with each node's contribution to the overall graph
                prediction, i.e., each atom's contribution to the overall
                molecular property prediction.
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns:
            A torch.FLoatTensor of size [num_graphs, num_outputs] with
            predictions for each graph (molecule).
        """
        num_graphs = torch.unique(graph_indexes).shape[0]

        atomic_contributions = atomic_contributions*self.scale + self.shift
        atomic_contributions = atomic_contributions + self.atom_refs(atoms)

        # Sum contributions for each graph
        output_per_graph = torch.zeros(
            (num_graphs, self.num_outputs),
            device=atomic_contributions.device,
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomic_contributions,
        )

        return output_per_graph