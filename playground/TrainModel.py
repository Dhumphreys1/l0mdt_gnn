import l0mdt_gnn
import yaml
import argparse
import os
import torch
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Process ROOT files and save to HDF5.")
parser.add_argument("--config", required=True, help="yaml config file describing inputs and model architecture")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Make output directories
output_dir = os.path.join(config['model_dir_base'], config['model_name'])
fig_path = os.path.join(output_dir, "figs")
event_display_path = os.path.join(fig_path, "event_displays")
os.makedirs(output_dir, exist_ok = True)
os.makedirs(event_display_path, exist_ok = True)

graphBuilder = l0mdt_gnn.graphBuilder(multilayer_scaler = config["multilayer_scaler"])
input_path = os.path.join(config["input_dir"], config["input_filename"])
station_inputs = dict(np.load(input_path, allow_pickle=True))

events = l0mdt_gnn.makeEvents(station_inputs)
start_time = time.time()
graphs, event_ids = l0mdt_gnn.getGraphs(events, graphBuilder)
end_time = time.time()
print(f"[INFO] Graph Building Time: {end_time - start_time:.4f} seconds for {len(graphs)} graphs.")
print(f"[INFO] Time/Graph: {(end_time - start_time)*1e6/len(graphs):.1f}us")

event_ids = np.array(event_ids)
test_mask = event_ids%10 == 0
val_mask = event_ids%10 == 1
train_mask = ~(test_mask | val_mask)

print("[INFO] Preparing test-train-val splits.")
train_graphs, train_ids = [g for i, g in enumerate(graphs) if train_mask[i]], event_ids[train_mask]
test_graphs, test_ids = [g for i, g in enumerate(graphs) if test_mask[i]], event_ids[test_mask]
val_graphs, val_ids = [g for i, g in enumerate(graphs) if val_mask[i]], event_ids[val_mask]
print("[INFO] Prepared test-train-val splits.")

print("[INFO] Calculating scalers.")
all_train_x = np.concatenate([g.x.detach().cpu().numpy() for g in train_graphs], axis=0)
all_train_edge_attr = np.concatenate([g.edge_attr.detach().cpu().numpy() for g in train_graphs], axis=0)

x_scaler = StandardScaler().fit(all_train_x)
edge_scaler = StandardScaler().fit(all_train_edge_attr)
with open(output_dir + "/x_scaler.pkl", "wb") as f:
    pickle.dump(x_scaler, f)
with open(output_dir + "/edge_scaler.pkl", "wb") as f:
    pickle.dump(edge_scaler, f)

print("[INFO] Scaling Data.")
l0mdt_gnn.scale_graphs(train_graphs, x_scaler, edge_scaler)
l0mdt_gnn.scale_graphs(test_graphs, x_scaler, edge_scaler)
l0mdt_gnn.scale_graphs(val_graphs, x_scaler, edge_scaler)
print("[INFO] Data scaled.")

total_pos = sum(graph.y.sum().item() for graph in train_graphs)
total_neg = sum((len(graph.y) - graph.y.sum().item()) for graph in train_graphs)
pos_weight_value = total_neg / total_pos
pos_weight = torch.tensor([pos_weight_value], device=device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("[INFO] Preparing dataloaders.")
np.random.seed(42)
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)
test_loader = DataLoader(test_graphs, batch_size=32)

print("[INFO] Building model.")
model = l0mdt_gnn.model(node_in = train_graphs[0].x.shape[1] ,
                        edge_in = train_graphs[0].edge_attr.shape[1],
                        hidden = 16,
                        out_dim = 1,
                        dropout = .2).to(device)


dummy_graph = test_graphs[0]
model_summary = summary(
    model,
    input_data=(dummy_graph.x.to(device), 
                dummy_graph.edge_index.to(device), 
                dummy_graph.edge_attr.to(device)),
    col_names=["num_params", "trainable"],
    verbose=0
)
print(f"[INFO] Total trainable parameters: {model_summary.trainable_params}")

optimizer = torch.optim.RAdam(model.parameters(), lr = config["learning_rate"])
print("[INFO] Beginning Training.")
# --- Training loop ---
history = {}
history["loss"] = []
history["val_loss"] = []
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        y_true = batch.y.to(device).float()

        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr).view(-1)
        loss = loss_fn(logits, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / sum(len(g.y) for g in train_graphs)
    history["loss"].append(avg_loss)
    

    # Optional: simple validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            y_true = batch.y.to(device).float()
            logits = model(x, edge_index, edge_attr).view(-1)
            val_loss += loss_fn(logits, y_true).item() * x.size(0)
        val_loss /= sum(len(g.y) for g in val_graphs)
        history["val_loss"].append(val_loss)
    print(f"Epoch {epoch+1}/{config['epochs']} - Train loss: {history['loss'][epoch]:.6f} - Val loss: {history['val_loss'][epoch]:.6f}")

epochs = np.linspace(1, config["epochs"], config["epochs"])
plt.figure(figsize=(6,6))
plt.plot(epochs, history["loss"], label = "Training Loss")
plt.plot(epochs, history["val_loss"], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.savefig(fig_path + f"/Loss_Curve.png")

model_path = os.path.join(output_dir, f"{config['model_name']}.pth")
torch.save(model.state_dict(), model_path)
# --- Inference on test set (node + graph-level) ---
model.eval()
all_pred_hits_per_graph = []
all_node_preds = []

with torch.no_grad():
    for batch in test_loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        
        logits = model(x, edge_index, edge_attr).view(-1)
        node_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_node_preds.append(node_pred)
        
        # Map node predictions back to graphs
        batch_index = batch.batch.cpu().numpy()
        num_graphs_in_batch = batch_index.max() + 1
        for g_idx in range(num_graphs_in_batch):
            node_pred_graph = node_pred[batch_index == g_idx]
            all_pred_hits_per_graph.append(node_pred_graph)

cut_off = 0.9
found_events = 0
total_events = 0
events_to_scan = np.linspace(0, 1000, 21, dtype=np.int32)
for event_num in events_to_scan:
    test_graph = test_graphs[event_num].to(device)
    if test_graph.x.size(0) == 0:
        continue

    with torch.no_grad():
        out = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)
        probs = torch.sigmoid(out).view(-1)
        preds = (probs > cut_off).cpu().numpy().astype(int)

    labels = test_graph.y.cpu().numpy().astype(int)

    # if labels.sum() > 0:
    #     continue

    tp = (preds == 1) & (labels == 1)
    tn = (preds == 0) & (labels == 0)
    fp = (preds == 1) & (labels == 0)
    fn = (preds == 0) & (labels == 1)

    if tp.sum() > labels.sum()-2:
        found_events += 1

    tp_mask = torch.tensor(tp, device=device)
    tn_mask = torch.tensor(tn, device=device)
    fp_mask = torch.tensor(fp, device=device)
    fn_mask = torch.tensor(fn, device=device)

    plt.figure(figsize=(14,8))
    plt.scatter(test_graph.x[tp_mask,0].cpu(), test_graph.x[tp_mask,1].cpu(), 
                marker='o', color='green', label='True Positive', s=40, edgecolors='k')
    plt.scatter(test_graph.x[tn_mask,0].cpu(), test_graph.x[tn_mask,1].cpu(), 
                marker='.', color='blue', label='True Negative', s=40)
    plt.scatter(test_graph.x[fp_mask,0].cpu(), test_graph.x[fp_mask,1].cpu(), 
                marker='x', color='red', label='False Positive', s=40)
    plt.scatter(test_graph.x[fn_mask,0].cpu(), test_graph.x[fn_mask,1].cpu(), 
                marker='x', color='black', label='False Negative', s=40)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Test Graph Predictions vs. True Labels, pred cutoff = {cut_off}, event: {event_num}')
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.show()
    plt.savefig(event_display_path + f"/{event_num}.png")
    plt.close()

cutoffs = np.linspace(0.0, 1.0, 100)
cutoffs, signal_eff, bkg_fake = l0mdt_gnn.event_level_sweep(test_graphs, model, device, min_hits_to_pass=3, cutoffs=cutoffs)

plt.figure(figsize=(6,6))
plt.plot(bkg_fake, signal_eff)
plt.xlabel("Background fake rate")
plt.ylabel("Signal efficiency")
plt.title("Event-level performance (≥3 hits)")
plt.grid(True)
plt.savefig(fig_path + "/ROC.png")

plt.figure(figsize=(6,6))
plt.plot(cutoffs, 100*bkg_fake, label = 'Bkg Efficiency')
plt.plot(cutoffs, 100*signal_eff, label = 'Signal Efficiency')
plt.legend(loc='upper right')
plt.xlabel("Score Threshold")
plt.ylabel("Efficiency %")
plt.title("Event-level performance (≥3 hits)")
plt.grid(True)
plt.savefig(fig_path + "/Efficiencies.png")


# # Optional: inspect node-level predictions for first test graph
# print("Node-level prediction for first test graph:")
# print(all_pred_hits_per_graph[0])
