# data_generator.py
import pandas as pd
import random
import pm4py
import os

# --- Import from project files ---
# This is needed for the stand-alone execution block
from config import CONFIG


class XESLogLoader:
    """
    Loads, processes, and prepares event logs from a collection of XES files.

    This class reads one or more XES files, builds a unified vocabulary for categorical
    attributes, and transforms the data into the nested list format required by the
    meta-learning framework. It is designed to be flexible and handles the absence
    of optional attributes like resources or costs gracefully.
    """

    def __init__(self):
        self.loaded_logs = {}
        self.vocab = {'activity': [], 'resource': []}
        self.activity_map = {}
        self.resource_map = {}

    def load_logs(self, log_paths: dict,
                  case_id_key='case:concept:name',
                  activity_key='concept:name',
                  timestamp_key='time:timestamp',
                  resource_key='org:resource',
                  cost_key='amount'):
        """
        Loads multiple XES logs, builds a unified vocabulary, and processes them.

        Args:
            log_paths (dict): A dictionary mapping a friendly name (e.g., 'A') to a file path.
            case_id_key (str): Column name for the case identifier.
            activity_key (str): Column name for the activity name.
            timestamp_key (str): Column name for the event timestamp.
            resource_key (str): Column name for the resource (optional).
            cost_key (str): Column name for a cost/amount attribute (optional).
        """
        print("Reading XES files and building unified vocabulary...")
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

        # --- Build Vocabulary ---
        # Mandatory attribute: Activity
        self.vocab['activity'] = sorted(list(combined_df[activity_key].unique()))
        self.activity_map = {name: i for i, name in enumerate(self.vocab['activity'])}

        # Robust handling of the Resource attribute
        resource_vocab_set = set()
        if resource_key in combined_df.columns:
            print(f"Found resource attribute '{resource_key}'.")
            combined_df[resource_key] = combined_df[resource_key].fillna('Unknown')
            resource_vocab_set.update(combined_df[resource_key].unique())
        else:
            print(f"⚠️ Warning: Resource attribute '{resource_key}' not found. Using a single placeholder.")

        # ALWAYS ensure 'Unknown' is in the vocabulary to serve as a reliable fallback.
        resource_vocab_set.add('Unknown')

        self.vocab['resource'] = sorted(list(resource_vocab_set))
        self.resource_map = {name: i for i, name in enumerate(self.vocab['resource'])}

        # Check for optional cost attribute
        if cost_key in combined_df.columns:
            print(f"Found cost attribute '{cost_key}'.")
        else:
            print(f"⚠️ Warning: Cost attribute '{cost_key}' not found. Random costs will be generated.")

        # --- Process each log individually using the unified vocabulary ---
        print("Transforming logs into framework-compatible format...")
        for name, group_df in combined_df.groupby('log_name'):
            self.loaded_logs[name] = self._convert_df_to_traces(
                group_df, case_id_key, activity_key, timestamp_key, resource_key, cost_key
            )
        print("✅ Log loading complete.")

    def _convert_df_to_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        """Converts a DataFrame into a list of traces with computed features."""
        processed_log = []

        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)
        df_grouped = df.groupby(case_id_key)

        for case_id, trace_df in df_grouped:
            trace_df = trace_df.sort_values(by=timestamp_key)
            if trace_df.empty:
                continue

            trace = []
            start_time = trace_df.iloc[0][timestamp_key]
            prev_time = start_time

            for _, event in trace_df.iterrows():
                current_time = event[timestamp_key]

                resource_name = event.get(resource_key, 'Unknown')
                resource_id = self.resource_map.get(resource_name, self.resource_map['Unknown'])

                cost_val = event.get(cost_key, round(random.uniform(5.0, 100.0), 2))
                if not isinstance(cost_val, (int, float)):
                    cost_val = round(random.uniform(5.0, 100.0), 2)

                event_dict = {
                    'case_id': case_id,
                    'activity': self.activity_map.get(event[activity_key]),
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
        """Returns the vocabulary sizes for model initialization."""
        return {
            'activity': len(self.vocab['activity']),
            'resource': len(self.vocab['resource']),
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
                label = (last_event_time - current_event_time) / 3600.0  # in hours
                tasks.append((prefix, label))

    return tasks


# --- Direct Execution Block (for demonstration) ---
if __name__ == '__main__':
    print("--- Demonstrating Flexible XES Log Loader ---")

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

            rev_activity_map = {v: k for k, v in loader.activity_map.items()}
            rev_resource_map = {v: k for k, v in loader.resource_map.items()}

            df['activity'] = df['activity'].map(rev_activity_map)
            df['resource'] = df['resource'].map(rev_resource_map)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['time_from_start'] = pd.to_timedelta(df['time_from_start'], unit='s')
            df['time_from_previous'] = pd.to_timedelta(df['time_from_previous'], unit='s')

            print("\n--- Sample of Processed Log 'A' (first 20 events) ---")
            print(df.head(20).to_string())

            print("\n--- Vocabulary Information ---")
            vocabs = loader.get_vocabs()
            print(f"Activity vocabulary size: {vocabs['activity']}")
            print(f"Resource vocabulary size: {vocabs['resource']}")
