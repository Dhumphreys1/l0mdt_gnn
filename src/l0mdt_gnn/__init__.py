import torch
import numpy as np
import matplotlib.pyplot as plt
from .GraphFactory_Jit import GraphBuilder
from .Classifier import SimpleEdgeGNN

# Add modules
graphBuilder = GraphBuilder
model = SimpleEdgeGNN


# Add standalone functions
def makeEvents(station_inputs):
    events = []
    for i in range(len(station_inputs['truth'])):
        event = {}
        for key in list(station_inputs.keys()):
            event[key] = station_inputs[key][i]
        events.append(event)
    return events

def getGraphs(events, graphBuilder):
    graphs = []
    event_ids = []
    for event in events:
        graph, event_number = graphBuilder.eventToGraph(event)
        if graph is None:
            continue
        graphs.append(graph)
        event_ids.append(event_number)
    return graphs, event_ids

def scale_graphs(graphs, node_scaler, edge_scaler):
    for graph in graphs:
        graph.x = torch.tensor(node_scaler.transform(graph.x.cpu().numpy()), dtype=torch.float)
        edge_attr_np = graph.edge_attr.cpu().numpy()
        if edge_attr_np.shape[0] > 0:  # only scale if non-empty
            graph.edge_attr = torch.tensor(edge_scaler.transform(edge_attr_np), dtype=torch.float)

def event_level_sweep(test_graphs, model, device, min_hits_to_pass=3, cutoffs=None):
    """
    Sweep node probability threshold and compute event-level efficiency and fake rate.

    Args:
        test_graphs: list of torch_geometric.Data objects
        model: trained node-level model
        device: 'cuda' or 'cpu'
        min_hits_to_pass: minimum number of predicted muon hits to consider event "passing"
        cutoffs: list or array of probability thresholds to evaluate

    Returns:
        cutoffs, signal_efficiency, background_fake_rate
    """
    if cutoffs is None:
        cutoffs = np.linspace(0.0, 1.0, 50)

    signal_efficiency = []
    background_fake_rate = []

    model.eval()
    with torch.no_grad():
        # Precompute node probabilities for all graphs
        all_node_probs = []
        all_true_hits = []
        for g in test_graphs:
            x, edge_index, edge_attr = g.x.to(device), g.edge_index.to(device), g.edge_attr.to(device)
            logits = model(x, edge_index, edge_attr).view(-1)
            node_probs = torch.sigmoid(logits).cpu().numpy()
            all_node_probs.append(node_probs)
            all_true_hits.append(g.y.cpu().numpy().astype(bool))

    for cut_off in cutoffs:
        n_signal_events = 0
        n_signal_lost = 0
        n_background_events = 0
        n_background_fakes = 0

        for node_probs, true_hits in zip(all_node_probs, all_true_hits):
            node_pred = node_probs > cut_off
            has_muons = true_hits.any()
            n_recovered = np.sum(node_pred & true_hits)
            n_false_pos = np.sum(node_pred & ~true_hits)

            if has_muons:
                n_signal_events += 1
                if n_recovered < min_hits_to_pass:
                    n_signal_lost += 1
            else:
                n_background_events += 1
                if n_false_pos >= min_hits_to_pass:
                    n_background_fakes += 1

        signal_efficiency.append((n_signal_events - n_signal_lost) / n_signal_events)
        background_fake_rate.append(n_background_fakes / n_background_events)

    return cutoffs, np.array(signal_efficiency), np.array(background_fake_rate)