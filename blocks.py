'''
This file contains the Message and Update classes. 

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
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    ds = torch.zeros(num_atoms, self.num_features).to(self.device)
    dvec = torch.zeros(num_atoms, 3, self.num_features).to(self.device)

    # Let S be the neighbors of the neigboring pairs in the egde index vector.
    # That is, S has the shape: num_edges x num_features (embedding)
    # We do the same for vec, which has the shape: num_edges x 3 x num_features (embedding)
    S = s[edge_indexes[1]]
    Vec = vec[edge_indexes[1]]

    # Atomwise layers.
    phi = self.linear_s(S)

    # Linear combination of the radial basis functions.
    edge_rbf_linear = self.linear_rbf(edge_rbf)

    # Cosine cutoff.
    fcut = CosineCutoff(edge_distance,
                        cutoff_dist)

    # Scale with fcut.
    W = edge_rbf_linear * fcut[..., None]

    # Split of W.
    vec_Ws, vec_Wvv, vec_Wvs = torch.split(phi * W, self.num_features, -1)

    # Aggregate contributions from neighboring atoms (scalar feature).
    ds = ds.index_add_(dim = 0,
                       index = edge_indexes[0],
                       source = vec_Ws,
                       alpha=1)

    # Standardize distance vector.
    vec_n = edge_vector / edge_distance[..., None]

    # Compute atomwise contribution. 
    dVec = vec_Wvv.unsqueeze(1) * Vec + vec_n.unsqueeze(2) * vec_Wvs.unsqueeze(1)

    # Aggregate contributions from neighboring atoms (vector feature).
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

        # Linear combinations of feature vector.
        vec_U, vec_V = torch.split(self.linear_vec(vec), self.num_features, dim = -1)

        # Compute dot product. 
        vec_dot = (vec_U * vec_V).sum(dim=1)

        # Vector norm. Add an epsilon offset to make sure sqrt is always positive.
        vec_Vn = torch.sqrt(torch.sum(vec_V**2, dim = -2) + 1e-8)

        # Concatinate with scalar feature and compute linear combinations.
        vec_W = self.linear_svec(torch.cat([s, vec_Vn], dim = -1))

        # Split vector.
        a_vv, a_sv, a_ss = torch.split(vec_W, self.num_features, dim = -1)

        # Compute final change in scalar feature.
        ds = a_ss + a_sv * vec_dot    # * self.inv_sqrt_2

        # Compute final change in vector feature.
        dvec = a_vv.unsqueeze(1) * vec_U

        return ds, dvec