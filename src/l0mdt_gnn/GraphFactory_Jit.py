import torch
import numpy as np
from numba import njit
from torch_geometric.data import Data


@njit(nopython=True)
def computeEdges(tube_number_arr, tube_layer_arr, eta_array, multilayer_arr, time_arr, charge_arr, adjacent_dist2 = 9, multilayer_scaler = 101):
    """
    Compute edges for a given set of hits.
    Edges are created between hits that are within a certain distance in the xy-plane,
    x and y have been discretized to row and column format.
    Inputs:
    -----------
    tube_number_arr = x coordinates of hits
    tube_layer_arr = y coordinates of hits
    eta_arry = eta station id
    multilayer_arr = multilayer of hit
    time_arr = time of hit
    charge_arr = charge of hit
    multilayer_scaler = distance threshold for hits in different multilayers
    """
    n_hits = len(tube_number_arr)
    max_edges = n_hits * (n_hits - 1) // 2

    src = np.empty(max_edges, dtype=np.int32)
    dst = np.empty(max_edges, dtype=np.int32)
    incoming_edges = np.zeros(n_hits, dtype=np.int32)
    outgoing_edges = np.zeros(n_hits, dtype=np.int32)
    edge_attr = np.empty((max_edges, 7), dtype=np.float32)
    edge_count = 0

    adjacent_dist2 = adjacent_dist2 # ds^2 = drow^2 + dcol^2 <= 5
    multilayer_dist2 = adjacent_dist2 + multilayer_scaler # ds^2 = drow^2 + dcol^2 + multilayer_scaler*dml^2 -> ds^2 = adjacent_dist2 + multilayer_scaler
    # scaler should be selected to be squared value to connect two layers in different multilayers accounting for angle. Roughly drow_max^2 + 1
    for i in range(n_hits):
      coli, rowi = tube_number_arr[i], tube_layer_arr[i]
      for j in range(i + 1, n_hits):
          colj, rowj = tube_number_arr[j], tube_layer_arr[j]
          dcol = tube_number_arr[j] - tube_number_arr[i]
          drow = tube_layer_arr[j] - tube_layer_arr[i]
          deta = eta_array[j] - eta_array[i]
          dt = time_arr[j] - time_arr[i]
          dcharge = charge_arr[j] - charge_arr[i]
          dmultilayer = abs(multilayer_arr[j] - multilayer_arr[i])

          dist2 = dcol * dcol + drow * drow
          if dmultilayer == 1:
              max_dist2 = multilayer_dist2
          else:
              max_dist2 = adjacent_dist2
          if dist2 < max_dist2:
            if (rowj > rowi) or ((rowj == rowi) and (colj > coli)):
              src[edge_count] = i
              dst[edge_count] = j
              outgoing_edges[i] += 1
              incoming_edges[j] += 1
              edge_attr[edge_count, 0] = dcol
              edge_attr[edge_count, 1] = drow
              edge_attr[edge_count, 2] = dist2
              edge_attr[edge_count, 3] = deta
              edge_attr[edge_count, 4] = dmultilayer
              edge_attr[edge_count, 5] = dt
              edge_attr[edge_count, 6] = dcharge
              edge_count += 1

    return (src[:edge_count], dst[:edge_count], edge_attr[:edge_count], incoming_edges, outgoing_edges)

class GraphBuilder():
    def __init__(self, adjacent_dist2=9, multilayer_scaler=101):
        self.adjacent_dist2 = adjacent_dist2
        self.multilayer_scaler = multilayer_scaler

    def eventToGraph(self, event):
        adjacent_dist2 = self.adjacent_dist2
        multilayer_scaler = self.multilayer_scaler
        (src, dst, edge_attr, incoming_edges, outgoing_edges) = computeEdges(
            event['tube_number'],
            event['tube_layer'],
            event['station_eta'],
            event['multilayer'],
            event['time'],
            event['charge'],
            adjacent_dist2 = adjacent_dist2,
            multilayer_scaler = multilayer_scaler
        )

         # Skip events with no edges
        if len(src) == 0:
            return None, None

        # these need to be torch tensors
        edge_index = torch.from_numpy(np.stack((src, dst), axis=0)).long()
        edge_attr = torch.from_numpy(edge_attr).float()
        incoming_edges = torch.tensor(incoming_edges, dtype=torch.long)
        outgoing_edges = torch.tensor(outgoing_edges, dtype=torch.long)

        features = np.vstack((event['tube_number'],
                              event['tube_layer'],
                              event['station_eta'],
                              event['multilayer'],
                              event['time'],
                              event['charge'],
                              incoming_edges,
                              outgoing_edges)).T
        y = torch.tensor(event['truth'].astype(np.bool_))
        x = torch.tensor(features, dtype=torch.float)
        #graph_label = torch.tensor([1.0 if y.sum() >= 3 else 0.0], dtype=torch.float)
        #data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_label=graph_label)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data, event["event_number"][0]
