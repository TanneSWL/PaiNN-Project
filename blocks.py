'''
This file contains the Message and Update classes

'''

import torch 
import torch.nn as nn
from utils import CosineCutoff

class MessageBlock(nn.Module):
  def __init__(self,
               num_features,
               num_rbf_features):
    super().__init__()

    self.num_features = num_features

    self.num_rbf_features = num_rbf_features

    self.linear_s = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.SiLU(),
        nn.Linear(num_features, num_features * 3),
        )

    self.linear_rbf = nn.Linear(num_rbf_features, num_features * 3)

  def forward(self,
              s,
              vec,
              edge_indexes,
              edge_vector,
              edge_distance,
              edge_rbf,
              cutoff_dist):

    # Compute number of atoms (nodes) in batch.
    num_atoms = s.size(0)

    # Initialize ds and dvec.
    ds = torch.zeros(num_atoms, self.num_features)
    dvec = torch.zeros(num_atoms, 3, self.num_features)

    # Let S be the neighbors of the neigboring pairs in the egde index vector.
    # That is, S has the shape: num_edges x num_features (embedding)
    # We do the same for vec, which has the shape: num_edges x 3 x num_features (embedding)
    S = s[edge_indexes[1]]
    Vec = vec[edge_indexes[1]]

    # Atomwise layers.
    phi = self.linear_s(S)

    # Compute radial basis functions.
    #edge_rbf = RadialBasis(edge_distance,
                           #self.num_features,
                           #cutoff_dist)

    # Linear combination of the radial basis functions.
    edge_rbf_linear = self.linear_rbf(edge_rbf)

    # Cosine cutoff.
    fcut = CosineCutoff(edge_distance,
                        cutoff_dist)

    W = edge_rbf_linear * fcut[..., None]

    # Split of W.
    vec_Ws, vec_Wvv, vec_Wvs = torch.split(phi * W, self.num_features, -1)

    # Aggregate contributions from neighboring atoms ?????
    ds = ds.index_add_(dim = 0,
                       index = edge_indexes[0],
                       source = vec_Ws,
                       alpha=1)

    vec_n = edge_vector / edge_distance[..., None]

    #dVec = vec_Wvv.unsqueeze(1) * Vec.unsqueeze(2) + vec_n * vec_Wvs.unsqueeze(1)
    #dVec = vec_Wvv * Vec + vec_n * vec_Wvs
    dVec = vec_Wvv.unsqueeze(1) * Vec + vec_n.unsqueeze(2) * vec_Wvs.unsqueeze(1)

    dvec = dvec.index_add_(dim = 0,
                           index = edge_indexes[0],
                           source = dVec,
                           alpha=1)

    return ds, dvec


class UpdateBlock(nn.Module):
    def __init__(self,
                 num_features):
        super().__init__()

        self.num_features = num_features

        self.linear_vec = nn.Linear(num_features, num_features * 2, bias=False)

        self.linear_svec = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )

    def forward(self,
                s,
                vec):

        vec_U, vec_V = torch.split(self.linear_vec(vec), self.num_features, dim = -1)

        vec_dot = (vec_U * vec_V).sum(dim=1) #* self.inv_sqrt_h

        vec_Vn = torch.sqrt(torch.sum(vec_V**2, dim = -2) + 1e-8)      # Add an epsilon offset to make sure sqrt is always positive.

        vec_W = self.linear_svec(torch.cat([s, vec_Vn], dim = -1))

        a_vv, a_sv, a_ss = torch.split(vec_W, self.num_features, dim = -1)

        ds = a_ss + a_sv * vec_dot    # * self.inv_sqrt_2

        dvec = a_vv.unsqueeze(1) * vec_U

        return ds, dvec