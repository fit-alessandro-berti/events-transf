# data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np

# --- Import from project files ---
# This is needed for the stand-alone execution block
from config import CONFIG


class XESLogLoader:
    """
    Loads and processes event logs from XES files.

    Instead of a global vocabulary, this class uses a fixed vocabulary size and
    assigns a NEW, RANDOM mapping of activities/resources to integer IDs for
    EVERY trace. This forces the model to rely on in-context learning to understand
    the process structure from the support set, rather than memorizing token meanings.
    """

    def __init__(self):
        self.loaded_logs = {}
        # No global vocabulary anymore. The size is fixed in the config.

    def load_logs(self, log_paths: dict,
                  case_id_key='case:concept:name',
                  activity_key='concept:name',
                  timestamp_key='time:timestamp',
                  resource_key='org:resource',
                  cost_key='amount'):
        """
        Loads multiple XES logs and processes them without a unified vocabulary.
        """
        print("Reading XES files...")
        all_dfs = []
        for name, path in log_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ Warning: File not found at {path}. Skipping.")
                continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                df['log_name'] = name  # Keep track of origin
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error reading or converting file {path}: {e}")
                continue

        if not all_dfs:
            print("❌ Error: No valid XES logs were loaded. Aborting.")
            return

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # --- No Vocabulary Building ---
        # The vocabulary is now defined by a fixed size in CONFIG.
        # Mappings are created randomly on-the-fly for each trace.
        print("Transforming logs with trace-level random token allocation...")
        for name, group_df in combined_df.groupby('log_name'):
            self.loaded_logs[name] = self._convert_df_to_traces(
                group_df, case_id_key, activity_key, timestamp_key, resource_key, cost_key
            )
        print("✅ Log loading complete.")

    def _convert_df_to_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        """
        Converts a DataFrame into a list of traces.
        For each trace, it creates a new random mapping from activity/resource names
        to a fixed set of integer IDs.
        """
        processed_log = []
        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)

        # Ensure resource and cost columns exist to avoid KeyErrors
        if resource_key not in df.columns:
            df[resource_key] = 'Unknown'
        else:
            df[resource_key] = df[resource_key].fillna('Unknown')

        if cost_key not in df.columns:
            # Create a placeholder cost column if it doesn't exist
            df[cost_key] = np.nan

        df_grouped = df.groupby(case_id_key)

        for case_id, trace_df in df_grouped:
            trace_df = trace_df.sort_values(by=timestamp_key)
            if trace_df.empty:
                continue

            # --- Per-Trace Random Mapping ---
            # 1. Get unique activities and resources in this specific trace
            unique_activities = trace_df[activity_key].unique()
            unique_resources = trace_df[resource_key].unique()

            # 2. Check if the number of unique items exceeds our fixed vocabulary
            if len(unique_activities) > CONFIG['fixed_activity_vocab_size']:
                unique_activities = unique_activities[:CONFIG['fixed_activity_vocab_size']]

            if len(unique_resources) > CONFIG['fixed_resource_vocab_size']:
                unique_resources = unique_resources[:CONFIG['fixed_resource_vocab_size']]

            # 3. Create a random mapping for this trace
            act_ids = random.sample(range(CONFIG['fixed_activity_vocab_size']), len(unique_activities))
            res_ids = random.sample(range(CONFIG['fixed_resource_vocab_size']), len(unique_resources))

            activity_map = {name: i for name, i in zip(unique_activities, act_ids)}
            resource_map = {name: i for name, i in zip(unique_resources, res_ids)}

            # --- Process Events in Trace ---
            trace = []
            start_time = trace_df.iloc[0][timestamp_key]
            prev_time = start_time

            for _, event in trace_df.iterrows():
                current_time = event[timestamp_key]

                # Use the trace-specific random map. Fallback to 0 if an item was truncated.
                activity_id = activity_map.get(event[activity_key], 0)
                resource_id = resource_map.get(event[resource_key], 0)

                cost_val = event.get(cost_key, round(random.uniform(5.0, 100.0), 2))
                if pd.isna(cost_val) or not isinstance(cost_val, (int, float)):
                    cost_val = round(random.uniform(5.0, 100.0), 2)

                event_dict = {
                    'case_id': case_id,
                    'activity': activity_id,
                    'timestamp': current_time.timestamp(),
                    'resource': resource_id,
                    'cost': cost_val,
                    'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time

            if trace:
                processed_log.append(trace)

        return processed_log

    def get_log(self, name: str):
        """Retrieves a processed log by its friendly name."""
        return self.loaded_logs.get(name)

    def get_vocabs(self):
        """Returns the FIXED vocabulary sizes for model initialization."""
        return {
            'activity': CONFIG['fixed_activity_vocab_size'],
            'resource': CONFIG['fixed_resource_vocab_size'],
        }


def get_task_data(log, task_type, max_seq_len=10):
    """
    Creates subsequences and corresponding labels for a given task.
    """
    tasks = []
    if not log:
        return tasks

    for trace in log:
        if len(trace) < 3: continue

        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            if task_type == 'classification':
                label = trace[i + 1]['activity']
                tasks.append((prefix, label))
            elif task_type == 'regression':
                last_event_time = trace[-1]['timestamp']
                current_event_time = prefix[-1]['timestamp']
                remaining_time_hours = (last_event_time - current_event_time) / 3600.0

                # FIX: Apply log transformation to the regression target for a more stable distribution.
                label = np.log1p(remaining_time_hours)

                tasks.append((prefix, label))

    return tasks


# --- Direct Execution Block (for demonstration) ---
if __name__ == '__main__':
    print("--- Demonstrating XES Log Loader with Random Per-Trace Tokenization ---")

    # --- MODIFIED: Load paths from the central CONFIG file ---
    all_paths = {**CONFIG['log_paths']['training'], **CONFIG['log_paths']['testing']}

    # Check if the 'logs' directory and files exist
    log_dir = './logs'
    if not os.path.isdir(log_dir) or not any(os.path.exists(p) for p in all_paths.values()):
        print("\n❌ CRITICAL: No XES files found in the './logs' directory.")
        print("Please create a 'logs' directory and add .xes files as defined in config.py to run this script.")
    else:
        # Run the XESLogLoader Demo
        loader = XESLogLoader()
        loader.load_logs(all_paths)

        # Display sample from a loaded log (e.g., log 'A')
        log_a_data = loader.get_log('A')
        if not log_a_data:
            print("\nLog 'A' could not be loaded. Please check the file path and format.")
        else:
            print(f"\nSuccessfully loaded log 'A' with {len(log_a_data)} traces.")

            # Prepare for display
            flat_log = [event for trace in log_a_data for event in trace]
            df = pd.DataFrame(flat_log)

            # Note: We can no longer display original activity/resource names because
            # the integer mapping is now random and local to each trace.
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['time_from_start'] = pd.to_timedelta(df['time_from_start'], unit='s')
            df['time_from_previous'] = pd.to_timedelta(df['time_from_previous'], unit='s')

            print("\n--- Sample of Processed Log 'A' (first 20 events) ---")
            print("Note: 'activity' and 'resource' are now random integer IDs, newly assigned for each trace.")
            print(df.head(20).to_string())

            print("\n--- Vocabulary Information ---")
            vocabs = loader.get_vocabs()
            print(f"Activity vocabulary size (fixed): {vocabs['activity']}")
            print(f"Resource vocabulary size (fixed): {vocabs['resource']}")
