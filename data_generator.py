# data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np

# --- Import from project files ---
# This is needed for the stand-alone execution block
from config import CONFIG
from time_transf import transform_time


class XESLogLoader:
    """
    Loads and processes event logs from XES files for dynamic re-mapping.

    This class reads XES files, stores the traces with their original categorical
    string values (e.g., 'register request'), and keeps track of the unique
    activities/resources for each log. A dedicated method, `remap_logs`, is
    used to apply a new random integer mapping to these raw traces, which is
    done at the start of each training epoch or test run.
    """

    def __init__(self):
        self.raw_logs = {}  # Stores logs with original string values
        self.processed_logs = {}  # Stores logs with integer mappings, refreshed each epoch
        self.log_specific_vocabs = {}  # Stores unique string values for each log, e.g., {'A': {'activity': [...]}}

    def load_logs(self, log_paths: dict,
                  case_id_key='case:concept:name',
                  activity_key='concept:name',
                  timestamp_key='time:timestamp',
                  resource_key='org:resource',
                  cost_key='amount'):
        """
        Loads multiple XES logs, extracts raw traces, and identifies log-specific vocabularies.
        No global vocabulary is built; mappings are handled by the `remap_logs` method.
        """
        print("Reading XES files and extracting raw trace data...")
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

        if cost_key not in combined_df.columns:
            print(f"⚠️ Warning: Cost attribute '{cost_key}' not found. Random costs will be generated.")

        # --- Process each log individually to get raw traces and vocabularies ---
        print("Extracting log-specific vocabularies and raw traces...")
        for name, group_df in combined_df.groupby('log_name'):
            # 1. Store unique string values for this specific log
            unique_activities = sorted(list(group_df[activity_key].unique()))

            unique_resources = set()
            if resource_key in group_df.columns:
                group_df[resource_key] = group_df[resource_key].fillna('Unknown')
                unique_resources.update(group_df[resource_key].unique())
            unique_resources.add('Unknown')  # Ensure 'Unknown' is always a fallback

            self.log_specific_vocabs[name] = {
                'activity': unique_activities,
                'resource': sorted(list(unique_resources))
            }

            # 2. Convert dataframe to raw traces (with string values for categoricals)
            self.raw_logs[name] = self._convert_df_to_raw_traces(
                group_df, case_id_key, activity_key, timestamp_key, resource_key, cost_key
            )
        print("✅ Log loading complete. Ready for dynamic mapping.")

    def remap_logs(self, fixed_vocab_sizes: dict):
        """Generates new random integer mappings for each log and applies them."""
        self.processed_logs.clear()  # Clear mappings from the previous epoch/run

        for name, vocabs in self.log_specific_vocabs.items():
            # Create random mapping for activities
            num_unique_acts = len(vocabs['activity'])
            if num_unique_acts > fixed_vocab_sizes['activity']:
                raise ValueError(
                    f"Log '{name}' has {num_unique_acts} activities, but fixed size is {fixed_vocab_sizes['activity']}.")
            act_targets = np.random.choice(fixed_vocab_sizes['activity'], num_unique_acts, replace=False)
            activity_map = {val: i for val, i in zip(vocabs['activity'], act_targets)}

            # Create random mapping for resources
            num_unique_res = len(vocabs['resource'])
            if num_unique_res > fixed_vocab_sizes['resource']:
                raise ValueError(
                    f"Log '{name}' has {num_unique_res} resources, but fixed size is {fixed_vocab_sizes['resource']}.")
            res_targets = np.random.choice(fixed_vocab_sizes['resource'], num_unique_res, replace=False)
            resource_map = {val: i for val, i in zip(vocabs['resource'], res_targets)}

            # Apply the new mapping to the raw log data
            log_with_new_ids = []
            if name not in self.raw_logs: continue

            for raw_trace in self.raw_logs[name]:
                processed_trace = []
                for event in raw_trace:
                    processed_event = event.copy()
                    processed_event['activity'] = activity_map.get(event['activity'])
                    processed_event['resource'] = resource_map.get(event['resource'])
                    processed_trace.append(processed_event)
                log_with_new_ids.append(processed_trace)

            self.processed_logs[name] = log_with_new_ids

    def _convert_df_to_raw_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        """Converts a DataFrame into a list of traces, keeping categorical values as strings."""
        raw_log = []
        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)

        for case_id, trace_df in df.groupby(case_id_key):
            trace_df = trace_df.sort_values(by=timestamp_key)
            if trace_df.empty: continue

            trace = []
            start_time = trace_df.iloc[0][timestamp_key]
            prev_time = start_time

            for _, event in trace_df.iterrows():
                current_time = event[timestamp_key]
                cost_val = event.get(cost_key, round(random.uniform(5.0, 100.0), 2))
                if not isinstance(cost_val, (int, float)):
                    cost_val = round(random.uniform(5.0, 100.0), 2)

                event_dict = {
                    'case_id': case_id,
                    'activity': event[activity_key],  # Store raw string
                    'timestamp': current_time.timestamp(),
                    'resource': event.get(resource_key, 'Unknown'),  # Store raw string
                    'cost': cost_val,
                    'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time

            if trace:
                raw_log.append(trace)
        return raw_log

    def get_log(self, name: str):
        """Retrieves a processed log by its name. Must be called after `remap_logs`."""
        return self.processed_logs.get(name)


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
                label = transform_time(remaining_time_hours)
                tasks.append((prefix, label))
    return tasks


# --- Direct Execution Block (for demonstration) ---
if __name__ == '__main__':
    print("--- Demonstrating Dynamic Mapping Log Loader ---")

    all_paths = {**CONFIG['log_paths']['training'], **CONFIG['log_paths']['testing']}

    if not os.path.isdir('./logs') or not any(os.path.exists(p) for p in all_paths.values()):
        print("\n❌ CRITICAL: No XES files found in the './logs' directory.")
    else:
        loader = XESLogLoader()
        loader.load_logs(all_paths)

        # Apply an initial random mapping to see the result
        print("\n🎲 Applying initial random mapping...")
        loader.remap_logs(CONFIG['fixed_vocab_sizes'])

        log_a_data = loader.get_log('A')
        if not log_a_data:
            print("\nLog 'A' could not be loaded or mapped.")
        else:
            print(f"\nSuccessfully loaded and mapped log 'A' with {len(log_a_data)} traces.")
            flat_log = [event for trace in log_a_data for event in trace]
            df = pd.DataFrame(flat_log)

            # Convert time columns for better readability
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['time_from_start'] = pd.to_timedelta(df['time_from_start'], unit='s')
            df['time_from_previous'] = pd.to_timedelta(df['time_from_previous'], unit='s')

            print("\n--- Sample of Processed Log 'A' with random integer IDs (first 20 events) ---")
            print(df[['case_id', 'activity', 'resource', 'cost', 'timestamp']].head(20).to_string())

            print("\n--- Vocabulary Information ---")
            vocabs = CONFIG['fixed_vocab_sizes']
            print(f"Activity vocabulary size (fixed): {vocabs['activity']}")
            print(f"Resource vocabulary size (fixed): {vocabs['resource']}")
