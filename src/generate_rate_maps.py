import sys
import argparse
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import timedelta
from scipy.ndimage import gaussian_filter

# --- Neural Data Imports ---
from pynwb import NWBHDF5IO # pyright: ignore[reportMissingImports]

# --- Graph Theory Imports ---
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# --- Helper Functions ---

def parse_video_to_seconds(ts_str):
    """Converts the HH:MM:SS.mmm string into 'Seconds From Creation'"""
    if not ts_str: return None
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
        return td.total_seconds()
    except Exception:
        return None

def compute_speed_from_xy(x: np.ndarray, y: np.ndarray, time_sec: np.ndarray) -> np.ndarray:
    if len(x) < 2: return np.zeros_like(x)
    dx = np.gradient(x)
    dy = np.gradient(y)
    dt = np.gradient(time_sec)
    dt[dt <= 0] = 1e-9 # Prevent division by zero
    spd = np.hypot(dx, dy) / dt
    return np.nan_to_num(spd, nan=0.0, posinf=0.0, neginf=0.0)

def build_hexmaze_graph(nodes_df):
    G = nx.Graph()
    nodes_df['id_str'] = nodes_df['id'].astype(int).astype(str)
    
    pos_dict = {}
    for idx, row in nodes_df.iterrows():
        node_id = row['id_str']
        G.add_node(node_id, pos=(row['x'], row['y']))
        pos_dict[node_id] = np.array([row['x'], row['y']])

    coords = nodes_df[['x', 'y']].values
    distances = squareform(pdist(coords))
    threshold = 65
    node_ids = nodes_df['id_str'].tolist()
    
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            dist = distances[i, j]
            if dist < threshold:
                G.add_edge(node_ids[i], node_ids[j], weight=dist)

    for n in ['501', '502']:
        if n in G: G.remove_node(n)

    manual_edges = [('121', '302'), ('324', '401'), ('305', '220'), 
                    ('404', '223'), ('201', '124'), ('224', '218')]
    
    for u, v in manual_edges:
        if u in G and v in G:
            p1 = pos_dict[u]
            p2 = pos_dict[v]
            w = np.linalg.norm(p1 - p2)
            G.add_edge(u, v, weight=w)
            
    return G

# --- Main Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Per-Trial Rate Maps for Neurons')
    parser.add_argument('-o', "--log_folder", required=True, help='Folder containing .log files')
    parser.add_argument('-n', "--nwb_file", required=True, help='Path to the .nwb sorting data file')
    parser.add_argument('-out', "--output_dir", default="neuron_trial_pdfs", help='Output folder for PDFs')
    
    args = parser.parse_args()

    log_dir = Path(args.log_folder)
    nwb_path = Path(args.nwb_file)
    out_dir = Path(args.output_dir)

    if not log_dir.exists(): sys.exit(f"Error: Log directory {log_dir} does not exist.")
    if not nwb_path.exists(): sys.exit(f"Error: NWB file {nwb_path} does not exist.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Neural Data ---
    print(f"Loading NWB file: {nwb_path}")
    try:
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            units_df = nwbfile.units.to_dataframe()
            good_df = units_df[units_df['quality'] == 'good'].reset_index(drop=True)
            good_units_spikes = good_df['spike_times'].tolist()
            unit_ids = good_df.index.tolist() if 'id' not in good_df.columns else good_df['id'].tolist()
            print(f"Found {len(good_units_spikes)} 'good' units.")
    except Exception as e:
        sys.exit(f"Error reading NWB file: {e}")

    # --- 2. Load and Parse Logs ---
    LOG_GLOB = str(log_dir / "*.log")
    log_paths = sorted(glob.glob(LOG_GLOB, recursive=True))
    if not log_paths: sys.exit(f"No .log files found in {log_dir}")

    ts_line_new = re.compile(
        r'^(?:(?P<level>[A-Z]+)\s*:\s*)?(?:(?P<video>\d{1,2}:\d{1,2}:\d{1,2}\.\d{3})\s*)?'
        r'(?:(?P<sys>\d+(?:\.\d+)?)\s*)?(?::\s*)?(?P<msg>.*)$'
    )
    pos_line = re.compile(r'The rat position is:\s*\(\s*(?P<x>-?[\d\.]+),\s*(?P<y>-?[\d\.]+)\s*\)\s*@\s*(?P<frame>[\d\.]+)')

    all_dfs = []
    print("Parsing logs for position and time (Seconds From Creation)...")
    for log_path in log_paths:
        rows_new = []
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                m = ts_line_new.match(line)
                if not m: continue
                
                # We extract the video time to use as "Seconds From Creation"
                vid_sec = parse_video_to_seconds(m.group("video"))
                msg = m.group("msg")
                
                x = y = None
                mpos = pos_line.search(msg)
                
                if mpos:
                    try:
                        x = int(float(mpos.group("x")))
                        y = int(float(mpos.group("y")))
                        event = "rat_position"
                    except ValueError:
                        continue
                elif msg.startswith("Recording Trial"): event = "recording_start"
                else: event = "message"
                
                rows_new.append({
                    "video_seconds": vid_sec, # THIS IS OUR SYNC TIME
                    "event": event, 
                    "x": x, "y": y, 
                    "raw": msg
                })
        if rows_new: all_dfs.append(pd.DataFrame(rows_new))

    df = pd.concat(all_dfs, ignore_index=True)
    df["video_seconds"] = pd.to_numeric(df["video_seconds"], errors="coerce")

    # Assign Trials
    trial_id_list = []
    current_trial = 1
    trial_re = re.compile(r"Recording\s*Trial\s*(\d+)\b", flags=re.I)
    for _, row in df.iterrows():
        if str(row.get("event", "")).lower() == "recording_start":
            m = trial_re.search(str(row.get("raw", "")))
            if m: current_trial = int(m.group(1))
        trial_id_list.append(current_trial)
    df["trial_id"] = trial_id_list

    # Filter and sort positions using our new sync time
    pos_df = df[df["event"] == "rat_position"].dropna(subset=["x", "y", "video_seconds"]).copy()
    pos_df = pos_df.sort_values(["trial_id", "video_seconds"])

    # --- 3. Scaling and Speed Calculation ---
    X_SCALE_DEN = (2352 / 2 / 9)
    Y_SCALE_DEN = (1424 / 2 / 5)
    
    pos_df['x_scaled'] = pos_df['x'] / X_SCALE_DEN
    pos_df['y_scaled'] = pos_df['y'] / Y_SCALE_DEN

    pos_df['speed'] = 0.0
    pos_df['dt'] = 0.0
    for tid, group in pos_df.groupby('trial_id'):
        x_s = group['x_scaled'].values
        y_s = group['y_scaled'].values
        t_s = group['video_seconds'].values
        
        speed = compute_speed_from_xy(x_s, y_s, t_s)
        dt = np.diff(t_s, prepend=t_s[0])
        dt[dt <= 0] = 1e-9 
        
        pos_df.loc[group.index, 'speed'] = speed
        pos_df.loc[group.index, 'dt'] = dt

    # --- 4. Load Nodes for Overlay ---
    node_file_path = Path("src/tools/node_list_new.csv")
    nodes_data = None
    if node_file_path.exists():
        nodes_df = pd.read_csv(node_file_path, header=None, names=["id", "x", "y"])
        nodes_df["x_scaled"] = nodes_df["x"] / X_SCALE_DEN
        nodes_df["y_scaled"] = nodes_df["y"] / Y_SCALE_DEN
        nodes_data = nodes_df

    # --- 5. Rate Map Parameters ---
    speed_threshold = 0.05 # Note: 0.05 m/s (because coordinates are scaled to meters)
    # 2.5 cm spatial bins (0.025 meters)
    bin_size = 0.025 
    
    # Create the bin edges
    bins_x = np.arange(0, 9 + bin_size, bin_size)
    bins_y = np.arange(0, 5 + bin_size, bin_size)
    
    sigma = 1.5

    # --- 6. Generate PDFs Per Unit ---
    print(f"Starting PDF generation for {len(good_units_spikes)} units...")

    for u_idx, (unit_id, unit_spikes) in enumerate(zip(unit_ids, good_units_spikes)):
        unit_spikes = np.array(unit_spikes)
        pdf_filename = out_dir / f"Unit_{unit_id}_trial_rate_maps.pdf"
        
        print(f"Processing Unit {unit_id} ({u_idx+1}/{len(unit_ids)}) -> {pdf_filename.name}")
        
        with PdfPages(pdf_filename) as pdf:
            # Loop through each trial
            for tid, trial_data in pos_df.groupby('trial_id'):
                if len(trial_data) < 10: continue 
                
                # --- Map Spikes to Frame Indexes (Searchsorted approach) ---
                t_seconds = trial_data['video_seconds'].values
                
                # Find the bounding timestamps for this trial
                t_start, t_end = t_seconds[0], t_seconds[-1]
                trial_spikes = unit_spikes[(unit_spikes >= t_start) & (unit_spikes <= t_end)]
                
                # Get exact indices for the spikes within the tracking array
                spike_indices = np.searchsorted(t_seconds, trial_spikes)
                
                # Ensure indices don't fall out of bounds
                valid_mask = (spike_indices > 0) & (spike_indices < len(t_seconds))
                spike_indices = spike_indices[valid_mask]

                # Extract X, Y, and Speed exactly at spike frames
                spike_x = trial_data['x_scaled'].values[spike_indices]
                spike_y = trial_data['y_scaled'].values[spike_indices]
                spike_spd = trial_data['speed'].values[spike_indices]

                # Filter spikes by speed
                valid_spd_mask = spike_spd > speed_threshold
                spike_x = spike_x[valid_spd_mask]
                spike_y = spike_y[valid_spd_mask]
                
                # --- Create Occupancy Map ---
                running_mask = trial_data['speed'] > speed_threshold
                running_pos = trial_data[running_mask]
                
                occupancy_map, _, _ = np.histogram2d(
                    running_pos['x_scaled'], 
                    running_pos['y_scaled'], 
                    bins=[bins_x, bins_y], 
                    weights=running_pos['dt']
                )
                occupancy_map[occupancy_map == 0] = np.nan
                
                # --- Create Spike Map ---
                if len(spike_x) > 0:
                    spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[bins_x, bins_y])
                else:
                    spike_map = np.zeros((len(bins_x)-1, len(bins_y)-1))

                # --- Calculate and Smooth Rate Map ---
                raw_rate_map = spike_map / occupancy_map
                valid_pixels = ~np.isnan(raw_rate_map)
                rate_map_filled = np.nan_to_num(raw_rate_map)
                
                smoothed_rate_map = gaussian_filter(rate_map_filled, sigma=sigma)
                smoothed_rate_map[~valid_pixels] = np.nan 
                
                # --- Plotting ---
                fig, ax = plt.subplots(figsize=(10, 6))
                
                cmap = plt.cm.jet
                cmap.set_bad(color='white')
                
                im = ax.pcolormesh(bins_x, bins_y, smoothed_rate_map.T, cmap=cmap, shading='auto')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Firing Rate (Hz)')
                
                # Overlay Hexmaze Nodes (if available)
                if nodes_data is not None:
                    ax.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                               s=60, facecolors='none', edgecolors='black', 
                               linewidths=1.5, alpha=0.5, zorder=5)
                    for _, nrow in nodes_data.iterrows():
                        ax.text(nrow["x_scaled"] + 0.1, nrow["y_scaled"], 
                                str(int(nrow["id"])), color='black', fontsize=5, 
                                va='center', zorder=6, alpha=0.5)

                ax.set_title(f'Unit {unit_id} - Trial {tid}\n(Speed > {speed_threshold} m/s, Spikes: {len(spike_x)})')
                ax.set_xlabel('X Position (Scaled)')
                ax.set_ylabel('Y Position (Scaled)')
                
                # Standardize bounds and aspect (Y is inverted here)
                ax.set_xlim(0, 9)
                ax.set_ylim(5, 0)
                ax.set_aspect('equal')
                
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nDone! All unit rate maps saved to the '{out_dir}' directory.")