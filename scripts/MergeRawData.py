import uproot as ur
import numpy as np
import glob

file_dir = "/data/dhumphreys/L0MDT/data_NGT/samples/root/"

#### Signal Tuple ####
file_name = "digiNtuple_updated_1M.root"
file_path = file_dir + file_name
tree_name = "MuonHitTest"
sig_branches = ['Digits_MDT_stationEta', 'Digits_MDT_stationIndex', 'Digits_MDT_stationPhi',
               'Digits_MDT_multiLayer', 'Digits_MDT_tube', 'Digits_MDT_tubeLayer',
               'Digits_MDT_globalPosX', 'Digits_MDT_globalPosY', 'Digits_MDT_globalPosZ',
               'Digits_MDT_charge', 'Digits_MDT_time', 'MdtSimHitsPdgId']
print(f"Loading signal tuple: {file_path}")
file = ur.open(file_path)
tree = file[tree_name]
signal_data = tree.arrays(sig_branches, library='np')
signal_data_processed = {}

for key in signal_data:
    if key != 'MdtSimHitsPdgId':
        signal_data_processed[key] = []

print("Applying chamber mask to signal tuple")
for k in range(len(signal_data['MdtSimHitsPdgId'])):
    eta_mask = signal_data['Digits_MDT_stationEta'][k] > 0
    phi_mask = signal_data['Digits_MDT_stationPhi'][k] == 3
    station_mask = signal_data['Digits_MDT_stationIndex'][k]%2 == 0
    muon_mask = abs(signal_data['MdtSimHitsPdgId'][k]) == 13
    mask = eta_mask & phi_mask & station_mask & muon_mask
    if mask.sum() >= 15:
        for key in signal_data_processed:
            signal_data_processed[key].append(signal_data[key][k][mask])

signal_data_processed['truth'] = []
for k in range(len(signal_data_processed['Digits_MDT_stationIndex'])):
    signal_data_processed['truth'].append(np.ones_like(signal_data_processed['Digits_MDT_stationIndex'][k]))

#### Bkg Tuple ####
combined_bkg = {}
bkg_branches = ['MdtChamHitId_stationEta', 'MdtChamHitId_stationIndex', 'MdtChamHitId_stationPhi', 
                'MdtChamHitId_multiLayer', 'MdtChamHitId_tube', 'MdtChamHitId_tubeLayer',
                'MdtChamHit_globalPosX', 'MdtChamHit_globalPosY', 'MdtChamHit_globalPosZ',
                'MdtChamHit_Adc', 'MdtChamHit_Tdc']
for key in bkg_branches:
    combined_bkg[key] = []

print("Loading bkg tuples")
for file_path in glob.iglob(file_dir + "ntuple_test_*.root", recursive=False):
    print(f"Loading: {file_path}")
# file_path = file_dir + file_name
    tree_name = 'BasicTesterTree'
    file = ur.open(file_path)
    tree = file[tree_name]
    data = tree.arrays(bkg_branches, library='np')
    for key in data:
        combined_bkg[key].append(np.array(data[key]))
#del data

for key in combined_bkg:
    combined_bkg[key] = np.concatenate([np.array(array) for array in combined_bkg[key]])

print("Applying chamber mask to bkg tuple")
for k in range(len(combined_bkg['MdtChamHit_Tdc'])):
    eta_mask = combined_bkg['MdtChamHitId_stationEta'][k] > 0
    phi_mask = combined_bkg['MdtChamHitId_stationPhi'][k] == 3
    station_mask = combined_bkg['MdtChamHitId_stationIndex'][k]%2 == 0
    mask = eta_mask & phi_mask & station_mask
    if mask.sum() > 0:
        for key in combined_bkg:
            combined_bkg[key][k] = combined_bkg[key][k][mask]
            
combined_bkg['truth'] = []
for k in range(len(combined_bkg['MdtChamHitId_stationIndex'])):
    combined_bkg['truth'].append(np.zeros_like(combined_bkg['MdtChamHitId_stationIndex'][k]))

sig_branches.pop()
sig_branches.append('truth')
bkg_branches.append('truth')
combined_branch_keys = ['station_eta', 'station_index', 'station_phi', 
                        'multilayer', 'tube_number', 'tube_layer',
                        'x', 'y', 'z', 'charge', 'time', 'truth']
print("Merging Files")
combined_data = {}
for k, key in enumerate(combined_branch_keys):
    combined_data[key] = []
    print(f"Merging key: {key}")
    for i in range(len(signal_data_processed['truth'])):
        if i == 100_000:
            break
        if i < 8000:
            combined_data[key].append(np.concatenate([signal_data_processed[sig_branches[k]][i], combined_bkg[bkg_branches[k]][i]]))
        else:
            combined_data[key].append(combined_bkg[bkg_branches[k]][i])

combined_data["event_number"] = []
for k in range(len(combined_data["truth"])):
    combined_data["event_number"].append(np.full_like(combined_data["truth"][k], k, dtype=np.int32))
    
station_0_inputs = {}
station_2_inputs = {}
station_4_inputs = {}
for key in combined_data:
    station_0_inputs[key] = []
    station_2_inputs[key] = []
    station_4_inputs[key] = []
for i in range(len(combined_data['station_index'])):
    mask_station_0 = combined_data['station_index'][i] == 0
    mask_station_2 = combined_data['station_index'][i] == 2
    mask_station_4 = combined_data['station_index'][i] == 4
    for key in combined_data:
        station_0_inputs[key].append(combined_data[key][i][mask_station_0])
        station_2_inputs[key].append(combined_data[key][i][mask_station_2])
        station_4_inputs[key].append(combined_data[key][i][mask_station_4])

for key in combined_data:
    station_0_inputs[key] = np.array(station_0_inputs[key], dtype=object)
    station_2_inputs[key] = np.array(station_2_inputs[key], dtype=object)
    station_4_inputs[key] = np.array(station_4_inputs[key], dtype=object)

print("Saving File: station_0_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_0_inputs.npz", **station_0_inputs)
print("Saving File: station_2_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_2_inputs.npz", **station_2_inputs)
print("Saving File: station_4_inputs.npz")
np.savez_compressed("/data/dhumphreys/L0MDT/data_NGT/samples/npy/station_4_inputs.npz", **station_4_inputs)
print("All Done!")