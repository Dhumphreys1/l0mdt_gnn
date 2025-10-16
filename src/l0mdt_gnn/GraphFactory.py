import numpy as np
from numba import njit
import torch
from torch_geometric.data import Data


@njit(nopython=True)
def computeEdges(x_arr, y_arr, layer_arr, tube_number_arr, eta_array, multilayer_arr, multilayer_dist2, adjacent_dist2):
    n_hits = len(x_arr)
    max_edges = .5 * n_hits * (n_hits - 1)

    src = np.empty(max_edges, dtype=np.int32)
    dst = np.empty(max_edges, dtype=np.int32)
    incoming_edges = np.zeros(n_hits, dtype=np.int32)
    outgoing_edges = np.zeros(n_hits, dtype=np.int32)
    edge_attr = np.empty((max_edges, 6), dtype=np.float32)

    edge_count = 0
    for i in range(n_hits):
      xi, yi = x_arr[i], y_arr[i]
      for j in range(i + 1, n_hits):
          xj, yj = x_arr[j], y_arr[j]

          dx = xj - xi
          dy = yj - yi
          dist2 = dx*dx + dy*dy

          dtube_layer = layer_arr[j] - layer_arr[i]
          dtube_number = tube_number_arr[j] - tube_number_arr[i]
          deta = eta_array[j] - eta_array[i]
          dmultilayer = abs(multilayer_arr[j] - multilayer_arr[i])

          if dmultilayer == 0:
              max_dist2 = adjacent_dist2
          else:
              max_dist2 = multilayer_dist2

          if dist2 < max_dist2:
            if (yj > yi) or ((yj == yi) and (xj > xi)):
              src[edge_count] = i
              dst[edge_count] = j
              outgoing_edges[i] += 1
              incoming_edges[j] += 1
              edge_attr[edge_count, 0] = dx
              edge_attr[edge_count, 1] = dy
              edge_attr[edge_count, 2] = dtube_layer
              edge_attr[edge_count, 3] = dtube_number
              edge_attr[edge_count, 4] = deta
              edge_attr[edge_count, 5] = dmultilayer
              edge_count += 1

    return (src[:edge_count], dst[:edge_count], edge_attr[:edge_count], incoming_edges, outgoing_edges)

#graph_inputs = np.load("/data/dhumphreys/L0MDT/for_julianne/graphInputs.npy", allow_pickle=True)
class GraphBuilder():
    # def __init__(config_file):
    #     self.edge_vars = config["edge_features"]
    #     self.node_vars = config["node_features"]

    def eventToGraph(self, event):
        multilayer_dist2 = 400*400
        adjacent_dist2 = 40*40
        (src, dst, edge_attr, incoming_edges, outgoing_edges) = computeEdges(x_arr = event['x'], y_arr = event['y'],
                                                                            layer_arr = event['tube_layer'],
                                                                            tube_number_arr = event['tube_number'],
                                                                            eta_array = event['station_eta'],
                                                                            multilayer_arr = event['multilayer'],
                                                                            multilayer_dist2 = multilayer_dist2,
                                                                            adjacent_dist2 = adjacent_dist2)
        # these need to be torch tensors
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        incoming_edges = torch.tensor(incoming_edges, dtype=torch.long)
        
        features = np.vstack((event['multilayer'],
                       event['station_eta'],
                       event['tube_layer'],
                       event['tube_number'],
                       event['x'],
                       event['y'],
                       event['time'],
                       event['charge'],
                       incoming_edges,
                       outgoing_edges)).T
        y = torch.tensorevent['truth'].astype(np.bool_)
        edge_index


    def event_to_graph(self, event):
        x_list = []
        y_list = []
        pos_list = []
        layer_list = []
        number_list = []
        for i in range(len(event['multilayer'])):
            multi_layer = event['multilayer'][i]
            station_eta = event['station_eta'][i]
            tube_layer = event['tube_layer'][i]
            tube_number = event['tube_number'][i]
            x_coord = event['x'][i]
            y_coord = event['y'][i]
            x_list.append([multi_layer, station_eta, tube_layer, tube_number, x_coord, y_coord])

            truth = bool(event['truth'][i])
            y_list.append(truth)

            pos_list.append([x_coord, y_coord])
            layer_list.append(tube_layer)
            number_list.append(tube_number)
        x = torch.tensor(x_list, dtype=torch.float)
        y = torch.tensor(y_list, dtype=torch.bool)
        pos = torch.tensor(pos_list, dtype=torch.long)

        multi_layer_spacing = 245
        edge_indices = [[], []]
        edge_attributes = []
        for j in range(len(pos_list)):
            for k in range(len(pos_list)):
                if k == j:
                    continue
                distance_squared = (pos_list[k][0] - pos_list[j][0]) ** 2 + (pos_list[k][1] - pos_list[j][1]) ** 2
                if distance_squared <= multi_layer_spacing ** 2:
                    edge_indices[0].append(k)
                    edge_indices[1].append(j)

                    diff_in_layer = abs(layer_list[k] - layer_list[j])
                    diff_in_number = abs(number_list[k] - number_list[j])
                    x_diff = abs(pos_list[k][0] - pos_list[j][0])
                    cosine = x_diff / np.sqrt(distance_squared)
                    edge_attributes.append([diff_in_layer, diff_in_number, cosine])

        if edge_indices[0]:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
        return data



