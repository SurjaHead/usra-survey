import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
# For TensorBoard event file reading
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuration
TENSORBOARD_LOG_DIR = Path("runs") # Assumes this script is in the root of the project
OUTPUT_FILE = Path("activation_analysis_results.txt")

def standardize_activation_name(name_str):
    """Converts common activation name variations to a canonical form."""
    name_lower = name_str.lower()
    # Prioritize direct match if the input is already canonical (e.g., "ReLU")
    # This also helps if an activation like "CustomAct" is used and isn't in the map.
    canonical_forms = {
        "ReLU", "ELU", "LeakyReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "Mish"
    }
    if name_str in canonical_forms:
        return name_str

    mapping = {
        "relu": "ReLU",
        "elu": "ELU",
        "leaky_relu": "LeakyReLU",
        "leakyrelu": "LeakyReLU",
        "gelu": "GELU",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "silu": "SiLU",
        "swish": "SiLU", # SiLU is also known as Swish
        "mish": "Mish",
        # Add other lowercase variations if they appear in folder names
    }
    return mapping.get(name_lower, name_str) # Fallback to original name_str if no lowercase mapping found

def parse_run_directory_name(dir_name_str):
    """
    Parses the directory name to extract model type and activation function.
    Handles two formats:
    1. MODELBASE_ACTIVATION_YYYYMMDD_HHMMSS (e.g., ResNet18_ReLU_20231027_123456)
    2. BERT_ACTIVATION (e.g., BERT_elu, BERT_leaky_relu)
    """
    parts = dir_name_str.split('_')
    model_type = None
    activation_name = None

    # Try parsing timestamped format first
    # MODELBASE_ACTIVATION_YYYYMMDD_HHMMSS
    if len(parts) >= 4:
        try:
            date_part = parts[-2]
            time_part = parts[-1]
            # Check if last two parts look like date (YYYYMMDD) and time (HHMMSS)
            if len(date_part) == 8 and date_part.isdigit() and \
               len(time_part) == 6 and time_part.isdigit():
                
                activation_name_candidate = parts[-3]
                model_type_parts = parts[:-3]
                if model_type_parts and activation_name_candidate: # Ensure these parts are not empty
                    model_type = "_".join(model_type_parts)
                    activation_name = activation_name_candidate
        except IndexError:
            # This error means parts[-1], parts[-2], or parts[-3] don't exist, which is fine,
            # it just means it's not the timestamped format. We'll try the BERT format next.
            pass 

    # If timestamped format didn't match or wasn't applicable, try BERT specific format
    if model_type is None and activation_name is None: # Proceed only if first parse failed
        if dir_name_str.upper().startswith("BERT_") and len(parts) >= 2:
            # Examples: BERT_elu -> parts = ['BERT', 'elu']
            #           BERT_leaky_relu -> parts = ['BERT', 'leaky', 'relu']
            # Model type is always BERT for this pattern
            model_type = parts[0] # Capture the exact casing from folder name e.g. BERT or Bert
            
            # Activation name is everything after the first underscore
            activation_name_candidate = "_".join(parts[1:])
            if activation_name_candidate: # Ensure there is something after "BERT_"
                activation_name = activation_name_candidate

    if model_type and activation_name:
        standardized_activation = standardize_activation_name(activation_name)
        # For debugging: 
        # print(f"Dir: '{dir_name_str}' -> Model: '{model_type}', Activation (raw): '{activation_name}', Activation (std): '{standardized_activation}'")
        return model_type, standardized_activation
    else:
        print(f"Warning: Directory name '{dir_name_str}' did not match known patterns (timestamped or BERT_activation). Skipping.")
        return None, None


def get_run_data(run_dir_path: Path):
    """
    Extracts model type, activation, average total activation time, and best validation accuracy
    from a single run directory.
    """
    model_type, activation_name = parse_run_directory_name(run_dir_path.name)
    if not model_type or not activation_name:
        return None

    # 1. Get accuracy from training_stats.json (priority)
    stats_file = run_dir_path / "training_stats.json"
    best_val_acc = None
    json_acc_found = False
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            current_best_acc = stats.get('best_val_acc', stats.get('final_val_acc'))
            if current_best_acc is not None:
                if isinstance(current_best_acc, str) and '%' in current_best_acc:
                    best_val_acc = float(current_best_acc.replace('%','')) / 100.0
                elif isinstance(current_best_acc, (int, float)):
                    best_val_acc = float(current_best_acc)
                else:
                    print(f"Warning: 'best_val_acc' in {stats_file} for run {run_dir_path.name} is not a recognizable number: {current_best_acc}. Will try TB.")
                    best_val_acc = None # Force trying TensorBoard
                
                if best_val_acc is not None:
                    json_acc_found = True # Mark that we found usable accuracy in JSON
            else:
                 print(f"Info: 'best_val_acc' or 'final_val_acc' not found in {stats_file} for {run_dir_path.name}. Will check TensorBoard for accuracy.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {stats_file} for run {run_dir_path.name}. Will check TensorBoard for accuracy.")
        except Exception as e:
            print(f"Warning: Error reading {stats_file} for run {run_dir_path.name}: {e}. Will check TensorBoard for accuracy.")
    else:
        if model_type.upper().startswith("BERT"):
            print(f"Info: {stats_file} not found for BERT run {run_dir_path.name}. Checking TensorBoard for accuracy.")
        # For non-BERT runs, it's a stronger warning if stats file is missing and was expected.
        # However, the logic will still proceed to check TB for activation times.

    # Initialize EventAccumulator for both activation time and potential TB accuracy fallback
    ea = None
    avg_activation_time = None
    tb_acc_found = False

    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        ea = EventAccumulator(str(run_dir_path)).Reload() # Load all data once

        # 2. Get average total activation time from TensorBoard events
        primary_activation_tag = 'Activation_Time/Total' # Used by helpers.py
        bert_specific_activation_tag = 'activation_time/total' # Used by bert.py
        
        # Check for BERT specific tag first if model is BERT
        if model_type.upper().startswith("BERT") and bert_specific_activation_tag in ea.Tags()['scalars']:
            activation_time_events = ea.Scalars(bert_specific_activation_tag)
            if activation_time_events:
                total_times = [event.value for event in activation_time_events]
                # Log all timings for MLP model regardless of activation type
                if model_type.startswith("MLP") and total_times:
                    print(f"Info: MLP Model ({run_dir_path.name} - {activation_name}) all activation times (BERT specific tag '{bert_specific_activation_tag}'): {total_times}")
                avg_activation_time = np.mean(total_times) if total_times else None
                if avg_activation_time is not None:
                    print(f"Info: Extracted avg_activation_time from BERT specific tag '{bert_specific_activation_tag}' for {run_dir_path.name}.")

        # If not found via BERT specific tag, or if not a BERT model, try the primary tag
        if avg_activation_time is None and primary_activation_tag in ea.Tags()['scalars']:
            activation_time_events = ea.Scalars(primary_activation_tag)
            if activation_time_events:
                total_times = [event.value for event in activation_time_events]
                # Log all timings for MLP model regardless of activation type
                if model_type.startswith("MLP") and total_times:
                    print(f"Info: MLP Model ({run_dir_path.name} - {activation_name}) all activation times (primary tag '{primary_activation_tag}'): {total_times}")
                avg_activation_time = np.mean(total_times) if total_times else None
                if avg_activation_time is not None and model_type.upper().startswith("BERT"):
                    print(f"Info: Extracted avg_activation_time from primary tag '{primary_activation_tag}' for BERT run {run_dir_path.name} (BERT specific tag not found/empty).")
        
        # Fallback for BERT activation time if still None (e.g. if bert_specific_activation_tag was not present)
        if avg_activation_time is None and model_type.upper().startswith("BERT"):
            # This block reuses the logic from previous attempt, but now it's a true fallback
            print(f"Info: Neither '{bert_specific_activation_tag}' nor '{primary_activation_tag}' yielded data for BERT run {run_dir_path.name}. Trying general alternative tags...")
            possible_bert_activation_time_tags = [
                'epoch_activation_time', 
                'ActivationTime/Epoch', 
                'activation_time_ms', 
                'Time/Activation',
                # bert_specific_activation_tag is already tried
                'Epoch Activation Time'
            ]
            bert_activation_time_tag_found = None
            for tag in possible_bert_activation_time_tags:
                if tag in ea.Tags()['scalars']:
                    bert_activation_time_tag_found = tag
                    break
            
            if bert_activation_time_tag_found:
                activation_time_events = ea.Scalars(bert_activation_time_tag_found)
                if activation_time_events:
                    total_times = [event.value for event in activation_time_events]
                    if total_times:
                        # Log all timings for MLP model regardless of activation type
                        if model_type.startswith("MLP"):
                            print(f"Info: MLP Model ({run_dir_path.name} - {activation_name}) all activation times (general alternative tag '{bert_activation_time_tag_found}'): {total_times}")
                        avg_activation_time = np.mean(total_times)
                        print(f"Info: Extracted avg_activation_time from general alternative tag '{bert_activation_time_tag_found}' for BERT run {run_dir_path.name}.")
                    else:
                        print(f"Warning: No data points for activation time tag '{bert_activation_time_tag_found}' in BERT run {run_dir_path.name}.")
                else:
                    print(f"Warning: Activation time tag '{bert_activation_time_tag_found}' found but no scalar events for BERT run {run_dir_path.name}.")
            else:
                print(f"Warning: Could not find 'Activation_Time/Total' or any known alternative activation time tag in TensorBoard for BERT run {run_dir_path.name}. Tried: {possible_bert_activation_time_tags}")
        
        # Final check for warnings if no activation time was found
        if avg_activation_time is None:
            if model_type.upper().startswith("BERT"):
                print(f"Warning: Could not find any known activation time tag for BERT run {run_dir_path.name}.")
            else:
                print(f"Warning: Scalar '{primary_activation_tag}' not found or empty in {run_dir_path.name}. Skipping activation time.")

        # 3. Fallback to TensorBoard for accuracy if not found in JSON, especially for BERT runs
        if not json_acc_found and model_type.upper().startswith("BERT"):
            # Try common accuracy tags
            possible_accuracy_tags = ['Acc/val', 'Accuracy/val', 'Val/Accuracy', 'val_accuracy', 'eval/accuracy']
            acc_tag_found = None
            for tag in possible_accuracy_tags:
                if tag in ea.Tags()['scalars']:
                    acc_tag_found = tag
                    break
            
            if acc_tag_found:
                accuracy_events = ea.Scalars(acc_tag_found)
                if accuracy_events:
                    accuracy_values = [event.value for event in accuracy_events]
                    if accuracy_values:
                        best_val_acc = float(np.max(accuracy_values)) # Take the max accuracy logged
                        tb_acc_found = True
                        print(f"Info: Extracted best_val_acc ({best_val_acc:.4f}) from TensorBoard tag '{acc_tag_found}' for BERT run {run_dir_path.name}.")
                    else:
                        print(f"Warning: No data points for accuracy tag '{acc_tag_found}' in BERT run {run_dir_path.name}.")
                else:
                     print(f"Warning: Accuracy tag '{acc_tag_found}' found but no scalar events for BERT run {run_dir_path.name}.")
            else:
                print(f"Warning: Could not find a known validation accuracy tag in TensorBoard for BERT run {run_dir_path.name}. Tried: {possible_accuracy_tags}")

    except Exception as e:
        if "corrupted" in str(e).lower() or "truncated" in str(e).lower():
             print(f"Warning: TensorBoard event file in {run_dir_path.name} might be corrupted or truncated: {e}. Problems fetching TB data.")
        else:
            print(f"Warning: Error processing TensorBoard data for {run_dir_path.name}: {e}. Problems fetching TB data.")
    finally:
        if ea and hasattr(ea, '_summary_reader') and hasattr(ea._summary_reader, 'close'):
             try:
                 ea._summary_reader.close()
             except Exception:
                 pass

    if avg_activation_time is None and best_val_acc is None:
        print(f"Info: No usable data (neither time nor accuracy) found for run {run_dir_path.name}. Skipping this run.")
        return None

    return {
        "model_type": model_type,
        "activation": activation_name,
        "avg_activation_time": avg_activation_time,
        "best_val_acc": best_val_acc
    }

def analyze_runs():
    all_run_data = []
    if not TENSORBOARD_LOG_DIR.exists() or not TENSORBOARD_LOG_DIR.is_dir():
        print(f"Error: TensorBoard log directory '{TENSORBOARD_LOG_DIR}' not found or is not a directory.")
        print(f"Please ensure this script is run from your project's root directory, or adjust TENSORBOARD_LOG_DIR.")
        return

    print(f"Scanning run directories in {TENSORBOARD_LOG_DIR}...")
    for item in TENSORBOARD_LOG_DIR.iterdir():
        if item.is_dir():
            data = get_run_data(item)
            if data:
                all_run_data.append(data)
    
    if not all_run_data:
        print("No valid run data found to analyze after scanning all subdirectories.")
        return

    data_by_model = defaultdict(list)
    for run in all_run_data:
        data_by_model[run["model_type"]].append(run)

    activation_tallies = defaultdict(lambda: {"top_5_speed": 0, "top_5_accuracy": 0})
    all_activations_ever_seen = set()

    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write(f"Activation Function Performance Analysis Report\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            # --- Individual Model Performance ---
            f.write("Individual Model Performance Details\n")
            f.write("-" * 70 + "\n")

            for model_type, runs in data_by_model.items():
                print(f"\nAnalyzing model type: {model_type} ({len(runs)} runs found)")
                f.write(f"\nModel: {model_type}\n")
                f.write("-" * 50 + "\n")
                
                for run in runs:
                    all_activations_ever_seen.add(run["activation"])

                # --- Top 5 for Speed (lower activation time is better) ---
                speed_runs = [r for r in runs if r["avg_activation_time"] is not None]
                if speed_runs:
                    speed_runs.sort(key=lambda x: x["avg_activation_time"]) # Ascending
                    
                    # Determine how many to show: all for MLP_MNIST, top 5 otherwise
                    num_to_show_speed = len(speed_runs) if model_type == "MLP_MNIST" else min(5, len(speed_runs))
                    
                    f.write(f"Top {num_to_show_speed} performers by speed (avg activation time, ms):\n")
                    header_speed = f"{'Activation':<15} | {'Avg Time (ms)':<15}\n"
                    f.write(header_speed)
                    f.write("-" * (len(header_speed) -1) + "\n")
                    print(f"  Top {num_to_show_speed} performers by speed (avg activation time, ms) for {model_type}:")
                    for i, run_data in enumerate(speed_runs[:num_to_show_speed]):
                        print(f"    {i+1}. {run_data['activation']:<10}: {run_data['avg_activation_time']:.3f} ms")
                        f.write(f"{run_data['activation']:<15} | {run_data['avg_activation_time']:<15.3f}\n")
                        if i < 5: # Only tally top 5 for overall summary
                            activation_tallies[run_data["activation"]]["top_5_speed"] += 1
                    f.write("\n")
                else:
                    print(f"  No activation time data available for {model_type} to rank speed.")
                    f.write("No activation time data available to rank speed.\n\n")

                # --- Top 5 for Accuracy (higher accuracy is better) ---
                accuracy_runs = [r for r in runs if r["best_val_acc"] is not None]
                if accuracy_runs:
                    accuracy_runs.sort(key=lambda x: x["best_val_acc"], reverse=True) # Descending

                    # Determine how many to show: all for MLP_MNIST, top 5 otherwise
                    num_to_show_accuracy = len(accuracy_runs) if model_type == "MLP_MNIST" else min(5, len(accuracy_runs))

                    f.write(f"Top {num_to_show_accuracy} performers by accuracy (best_val_acc):\n")
                    header_acc = f"{'Activation':<15} | {'Best Val Acc':<15}\n"
                    f.write(header_acc)
                    f.write("-" * (len(header_acc) -1) + "\n")
                    print(f"  Top {num_to_show_accuracy} performers by accuracy (best_val_acc) for {model_type}:")
                    for i, run_data in enumerate(accuracy_runs[:num_to_show_accuracy]):
                        acc_display_val = run_data['best_val_acc']
                        acc_display_str = f"{acc_display_val:.4f}" if acc_display_val <= 1.0 else f"{acc_display_val:.2f}%"
                        print(f"    {i+1}. {run_data['activation']:<10}: {acc_display_str}")
                        f.write(f"{run_data['activation']:<15} | {acc_display_str:<15}\n")
                        if i < 5: # Only tally top 5 for overall summary
                            activation_tallies[run_data["activation"]]["top_5_accuracy"] += 1
                    f.write("\n")
                else:
                    print(f"  No accuracy data available for {model_type} to rank accuracy.")
                    f.write("No accuracy data available to rank accuracy.\n\n")
                f.write("=" * 50 + "\n") # Separator for next model

            # --- Overall Summary Table ---
            f.write("\n\nOverall Activation Function Performance Summary\n")
            f.write("=" * 70 + "\n")
            f.write("This table shows how many times each activation function ranked in the top 5 \n")
            f.write("for speed (lowest average activation time) and accuracy (highest validation accuracy)\n")
            f.write("across all analyzed model types.\n\n")
            
            summary_header = f"{'Activation':<15} | {'Top 5 Speed Tally':<20} | {'Top 5 Accuracy Tally':<22}\n"
            f.write(summary_header)
            f.write("-" * (len(summary_header) -1) + "\n")
            
            sorted_activations = sorted(list(all_activations_ever_seen))
            
            # Data structure for combined tallies
            combined_tallies_list = []

            if not sorted_activations:
                 f.write("No activation data found to summarize.\n")
            else:
                for activation_name in sorted_activations:
                    tallies = activation_tallies[activation_name]
                    f.write(
                        f"{activation_name:<15} | "
                        f"{tallies['top_5_speed']:<20} | "
                        f"{tallies['top_5_accuracy']:<22}\n"
                    )
                    # Store for combined table
                    total_mentions = tallies['top_5_speed'] + tallies['top_5_accuracy']
                    combined_tallies_list.append((activation_name, total_mentions))

            # --- New Combined Tally Table ---
            f.write("\n\nCombined Top 5 Performance Ranking\n")
            f.write("=" * 70 + "\n")
            f.write("This table sums the 'Top 5 Speed Tally' and 'Top 5 Accuracy Tally' \n")
            f.write("to give an overall ranking based on total top 5 mentions.\n\n")

            combined_header = f"{'Activation':<15} | {'Total Top 5 Mentions':<25}\n"
            f.write(combined_header)
            f.write("-" * (len(combined_header) -1) + "\n")

            # Sort by total mentions, descending
            combined_tallies_list.sort(key=lambda x: x[1], reverse=True)

            if not combined_tallies_list:
                f.write("No data for combined ranking.\n")
            else:
                for activation_name, total_mentions in combined_tallies_list:
                    f.write(f"{activation_name:<15} | {total_mentions:<25}\n")

            f.write("\nReport End\n")
        print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error: Could not write to output file {OUTPUT_FILE}. {e}")


if __name__ == "__main__":
    # Import datetime for the report
    from datetime import datetime
    analyze_runs()