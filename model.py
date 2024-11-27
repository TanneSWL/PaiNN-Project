'''
This file contains the paiNN-model itself 

'''

import torch 
import torch.nn as nn 
from blocks import MessageBlock, UpdateBlock
from utils import LocalEdges, RadialBasis

class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
        device: str = 'cpu'
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether
                two nodes (atoms) are neighbours.
        """
        super().__init__()

        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist
        self.device = device

        self.embedding_s = nn.Embedding(num_unique_atoms, num_features)
        

        self.message = nn.ModuleList()
        self.update = nn.ModuleList()

        for i in range(num_message_passing_layers): #Loop for number of messagepassing layers
            self.message.append(MessageBlock(num_features, num_rbf_features))
            self.update.append(UpdateBlock(num_features))


        self.output = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Linear(num_features // 2, 1),
        )

    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        # ----------------------------------------------------------------------
        # EMBEDDING
        # We initialize learnable embeddings for the atomtype.
        # The directions v_i are embedded by a null vector.

        s = self.embedding_s(atoms)
        vec = torch.zeros(s.size(0), 3, s.size(1)).to(self.device)

        # ----------------------------------------------------------------------
        # LOCAL NEIGHBORHOOD
        # We create edges by the relative position of nodes from a specified
        # cutoff within the same molecule (local interactions)

        edge_indexes, edge_vector, edge_distance = LocalEdges(atom_positions,
                                                              graph_indexes,
                                                              self.cutoff_dist)
        
        edge_indexes = edge_indexes.to(self.device)
        edge_vector = edge_vector.to(self.device)
        edge_distance = edge_distance.to(self.device)

        # ----------------------------------------------------------------------
        # RADIAL BASIS

        edge_rbf = RadialBasis(edge_distance,
                               self.num_rbf_features,
                               self.cutoff_dist)
        edge_rbf = edge_rbf.to(self.device)
        # ----------------------------------------------------------------------
        # MESSAGE AND UPDATE

        for i in range(self.num_message_passing_layers):
            ds, dvec = self.message[i](s, vec, edge_indexes, edge_vector, edge_distance, edge_rbf, self.cutoff_dist)
            s = s + ds
            vec = vec + dvec

            ds, dvec = self.update[i](s, vec)
            s = s + ds
            vec = vec + dvec

        # ----------------------------------------------------------------------
        # ATOMIC CONTRIBUTIONS

        atomic_contributions = self.output(s)

        # ----------------------------------------------------------------------

        # Final output
        return atomic_contributions