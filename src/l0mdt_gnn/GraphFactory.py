import numpy as np
import torch
from torch_geometric.data import Data

graph_inputs = np.load("/data/dhumphreys/L0MDT/for_julianne/graphInputs.npy", allow_pickle=True)

def event_to_graph(event):
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

event_to_graph(graph_inputs[0])

def events_to_graph(events):
    return [event_to_graph(event) for event in events]

events_to_graph(graph_inputs[:3])