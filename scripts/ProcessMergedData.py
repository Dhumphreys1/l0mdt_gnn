import numpy as np

station_0_tube_map = np.array([0, 30, 36, 36, 36, 30], dtype = np.int32)
station_2_tube_map = np.array([0, 48, 56, 56, 40, 40], dtype = np.int32)
station_4_tube_map = np.array([0, 64, 72, 56, 72, 72], dtype = np.int32)

# Helper function to cast arrays
def recast_field(data, key):
    data[key] = np.array([arr.astype(np.int32) for arr in data[key]], dtype=object)

def ProcessInputs(inputs, station_id):
    nEvents = len(inputs['truth'])
    if station_id == 0:
        tube_map = station_0_tube_map
        layers_per_ml = 4
    elif station_id == 2:
        tube_map = station_2_tube_map
        layers_per_ml = 3
    elif station_id == 4:
        tube_map = station_4_tube_map
        layers_per_ml = 3
        
    for i in range(nEvents):
        inputs["tube_number"][i] += tube_map[inputs["station_eta"][i]-1]
        inputs["tube_layer"][i] += layers_per_ml*(inputs["multilayer"][i]-1)

for i in range(1, len(station_0_tube_map)):
    station_0_tube_map[i] = station_0_tube_map[i-1] + station_0_tube_map[i]
    station_2_tube_map[i] = station_2_tube_map[i-1] + station_2_tube_map[i]
    station_4_tube_map[i] = station_4_tube_map[i-1] + station_4_tube_map[i]

station_0_inputs = dict(np.load("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_0_inputs.npz", allow_pickle=True))
station_2_inputs = dict(np.load("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_2_inputs.npz", allow_pickle=True))
station_4_inputs = dict(np.load("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_4_inputs.npz", allow_pickle=True))

# Recast both tube_number and tube_layer
for d in [station_0_inputs, station_2_inputs, station_4_inputs]:
    recast_field(d, "tube_number")
    recast_field(d, "tube_layer")

ProcessInputs(station_0_inputs, 0)
print("Done 0")
ProcessInputs(station_2_inputs, 2)
print("Done 2")
ProcessInputs(station_4_inputs, 4)
print("Done 4")

print("Saving File: station_0_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_0_inputs.npz", **station_0_inputs)
print("Saving File: station_2_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_2_inputs.npz", **station_2_inputs)
print("Saving File: station_4_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_4_inputs.npz", **station_4_inputs)
print("All Done!")