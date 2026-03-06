import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from pynwb import NWBHDF5IO # pyright: ignore[reportMissingImports]

def generate_unit_reports(nwb_path, coords_path, seconds_path, output_pdf_path, 
                          bin_size=5, speed_threshold=10, sigma=1.5):
    """
    Generates a multipage PDF where each page contains 4 plots for a single unit:
    1. Spike Scatter Plot
    2. Occupancy Map
    3. Speed Map
    4. Smoothed Rate Map
    """
    print("1. Loading Data for Overview Maps...")
    
    # --- Load NWB Data ---
    good_units_spikes = []
    unit_ids = []
    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        good_df = units_df[units_df['quality'] == 'good']
        good_units_spikes = good_df['spike_times'].tolist()
        unit_ids = good_df.index.tolist() if 'id' not in good_df.columns else good_df['id'].tolist()
    
    print(f"   -> Found {len(unit_ids)} 'good' units.")

    # --- Load Tracking Data ---
    df_coords = pd.read_csv(coords_path)
    df_seconds = pd.read_csv(seconds_path)
    merged_df = pd.concat([df_seconds, df_coords], axis=1)
    
    pos = merged_df[['Seconds From Creation', 'Rat_X', 'Rat_Y']]
    pos = pos[pos['Seconds From Creation'] > 0].reset_index(drop=True)
    
    # --- Calculate Speed and dt ---
    print("2. Computing Speed and Global Maps...")
    dx = np.diff(pos['Rat_X'], prepend=pos['Rat_X'].iloc[0])
    dy = np.diff(pos['Rat_Y'], prepend=pos['Rat_Y'].iloc[0])
    dt = np.diff(pos['Seconds From Creation'], prepend=pos['Seconds From Creation'].iloc[0])
    dt[dt <= 0] = 1e-9 # Prevent division by zero
    
    pos['Speed'] = np.sqrt(dx**2 + dy**2) / dt
    
    # --- Define Bins ---
    x_min, x_max = pos['Rat_X'].min(), pos['Rat_X'].max()
    y_min, y_max = pos['Rat_Y'].min(), pos['Rat_Y'].max()
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    # --- Compute Global Maps (Occupancy & Speed) ---
    running_mask = pos['Speed'] > speed_threshold
    running_pos = pos[running_mask]
    running_dt = dt[running_mask]

    # 1. Occupancy Map
    occupancy_map, _, _ = np.histogram2d(
        running_pos['Rat_X'], running_pos['Rat_Y'], 
        bins=[x_bins, y_bins], weights=running_dt
    )
    
    # 2. Speed Map
    speed_sum_map, _, _ = np.histogram2d(
        running_pos['Rat_X'], running_pos['Rat_Y'], 
        bins=[x_bins, y_bins], weights=running_pos['Speed'] * running_dt
    )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_speed_map = speed_sum_map / occupancy_map
        avg_speed_map[occupancy_map == 0] = np.nan 

    occupancy_map[occupancy_map == 0] = np.nan

    # --- Generate PDF ---
    print(f"3. Generating PDF: {output_pdf_path}")
    os.makedirs(os.path.dirname(output_pdf_path) or '.', exist_ok=True)
    
    cmap = plt.cm.jet
    cmap.set_bad(color='white')

    with PdfPages(output_pdf_path) as pdf:
        for u_idx, (unit_id, unit_spikes) in enumerate(zip(unit_ids, good_units_spikes)):
            print(f"   -> Processing Unit {unit_id} ({u_idx + 1}/{len(unit_ids)})")
            
            # --- Match Spikes to Positions ---
            spike_indices = np.searchsorted(pos['Seconds From Creation'], unit_spikes)
            valid_idx_mask = (spike_indices > 0) & (spike_indices < len(pos))
            spike_indices = spike_indices[valid_idx_mask]
            
            current_spike_x = pos.iloc[spike_indices]['Rat_X'].values
            current_spike_y = pos.iloc[spike_indices]['Rat_Y'].values
            current_spike_speed = pos.iloc[spike_indices]['Speed'].values
            
            # Filter spikes by speed
            valid_speed_mask = current_spike_speed > speed_threshold
            filtered_spike_x = current_spike_x[valid_speed_mask]
            filtered_spike_y = current_spike_y[valid_speed_mask]
            
            # --- Calculate Spike and Rate Maps ---
            spike_map, _, _ = np.histogram2d(filtered_spike_x, filtered_spike_y, bins=[x_bins, y_bins])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_rate_map = spike_map / occupancy_map
                
            valid_pixels = ~np.isnan(raw_rate_map)
            rate_map_filled = np.nan_to_num(raw_rate_map)
            
            smoothed_rate_map = gaussian_filter(rate_map_filled, sigma=sigma)
            smoothed_rate_map[~valid_pixels] = np.nan
            
            peak_rate = np.nanmax(smoothed_rate_map) if np.any(valid_pixels) else 0.0

            # --- Plotting 2x2 Grid ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"Unit {unit_id} Overview (Speed > {speed_threshold})\nSpikes: {len(filtered_spike_x)} | Peak Rate: {peak_rate:.2f} Hz", fontsize=16)
            
            # 1. Scatter Plot
            ax_scatter = axes[0, 0]
            ax_scatter.plot(pos['Rat_X'], pos['Rat_Y'], color='lightgrey', alpha=0.5, label='Trajectory')
            ax_scatter.scatter(filtered_spike_x, filtered_spike_y, color='red', s=4, label='Spikes', zorder=5)
            ax_scatter.set_title("Spike Scatter Plot")
            ax_scatter.legend(loc='upper right')
            
            # 2. Occupancy Map
            ax_occ = axes[0, 1]
            im_occ = ax_occ.pcolormesh(x_bins, y_bins, occupancy_map.T, cmap=cmap, shading='auto')
            fig.colorbar(im_occ, ax=ax_occ, label='Time (Seconds)')
            ax_occ.set_title("Occupancy Map")
            
            # 3. Speed Map
            ax_spd = axes[1, 0]
            im_spd = ax_spd.pcolormesh(x_bins, y_bins, avg_speed_map.T, cmap=cmap, shading='auto')
            fig.colorbar(im_spd, ax=ax_spd, label='Average Speed')
            ax_spd.set_title("Average Speed Map")
            
            # 4. Rate Map
            ax_rate = axes[1, 1]
            im_rate = ax_rate.pcolormesh(x_bins, y_bins, smoothed_rate_map.T, cmap=cmap, shading='auto')
            fig.colorbar(im_rate, ax=ax_rate, label='Firing Rate (Hz)')
            ax_rate.set_title("Smoothed Rate Map")
            
            for ax in axes.flatten():
                ax.set_aspect('equal')
                ax.invert_yaxis() 
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            pdf.savefig(fig)
            plt.close(fig) 

    print("Done! Overview PDF generated successfully.")

# ==========================================
# --- Command Line Interface ---
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 4-Panel Unit Overviews')
    parser.add_argument('-n', '--nwb_file', required=True, help='Path to NWB file')
    parser.add_argument('-c', '--coords_file', required=True, help='Path to Coordinates CSV')
    parser.add_argument('-s', '--seconds_file', required=True, help='Path to Seconds CSV')
    parser.add_argument('-out', '--output_pdf', required=True, help='Output PDF path')
    
    args = parser.parse_args()
    
    generate_unit_reports(
        nwb_path=args.nwb_file, 
        coords_path=args.coords_file, 
        seconds_path=args.seconds_file, 
        output_pdf_path=args.output_pdf,
        bin_size=5,             
        speed_threshold=10,     
        sigma=1.5               
    )