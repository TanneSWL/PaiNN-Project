'''
This file contains helper functions.

Functions: 

LocalEdges   - defines neighboring atoms within a molecule.
RadialBasis  - computes radial basis functions for the distance between two atoms. 
CosineCutoff - computes cosine cutoff.
'''

import torch

def LocalEdges(atom_positions,
               graph_indexes,
               cutoff_dist):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # The number of atoms in the batch.
    num_atoms = graph_indexes.size(0)

    # Pairing of the atoms across all molecules in the batch.
    pos_i = atom_positions.unsqueeze(0).repeat(num_atoms, 1, 1)
    pos_j = atom_positions.unsqueeze(1).repeat(1, num_atoms, 1)

    # Compute all r_ij vectors and their norms (distances).
    r_ij = pos_j - pos_i                                                    # Pairwise vector differences.
    r_ij = r_ij.to(device)
    r_ij_norm = torch.norm(r_ij, dim=2).to(device)                          # Pairwise distances.

    # We will not consider the distance between an atom and itself (i == j).
    # We only consider atoms within the same molecule (graph_indexes[i] == graph_indexes[j]).
    # We only want the pairs of close atoms specified by the cutoff.
    # Thus, we create masks to filter pairs.
    same_graph_mask     = graph_indexes.unsqueeze(0) == graph_indexes.unsqueeze(1)
    different_atom_mask = torch.arange(num_atoms).unsqueeze(1) != torch.arange(num_atoms).unsqueeze(0)
    within_cutoff_mask  = r_ij_norm <= cutoff_dist

    # Add all to device to ensure same device
    same_graph_mask = same_graph_mask.to(device)
    different_atom_mask = different_atom_mask.to(device)
    within_cutoff_mask = within_cutoff_mask.to(device)

    # Combine masks: same graph, different atoms, within cutoff.
    valid_pairs_mask = same_graph_mask & different_atom_mask & within_cutoff_mask

    # Filter indices and values based on the mask.
    edge_indexes = valid_pairs_mask.nonzero(as_tuple=False).t()   # Edge indexes, shape: (2, num_edges) - nonzero returns the indices of the elements that are non-zero (False is interpreted as 0).
    edge_vector = r_ij[valid_pairs_mask]                          # Edge vectors, shape: (num_edges, 3)
    edge_distance = r_ij_norm[valid_pairs_mask]                   # Edge distances, shape: (num_edges, 1)

    return edge_indexes, edge_vector, edge_distance

def RadialBasis(edge_distance,
                num_rbf_features,
                cutoff_dist):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Number of local edges.
    num_egdes = edge_distance.size()[0]

    # Generate n values evenly spaced between 1 and 20.
    n_values = torch.linspace(1, 20, num_rbf_features)

    # Expand the n_values to match the shape of edge_distance.
    n_values_expanded = n_values.unsqueeze(0).expand(num_egdes, num_rbf_features).to(device)
    edge_distance_expanded = edge_distance.unsqueeze(1).expand(num_egdes, num_rbf_features).to(device)

    # Compute the RBF for each pair of (r_ij, n).
    edge_rbf = torch.sin(n_values_expanded * torch.pi * edge_distance_expanded / cutoff_dist) / edge_distance_expanded

    return edge_rbf

def CosineCutoff(edge_distance,
                 cutoff_dist):

    # Compute values of cutoff function.
    fcut = 0.5 * (torch.cos(edge_distance * torch.pi / cutoff_dist) + 1.0)

    return fcut