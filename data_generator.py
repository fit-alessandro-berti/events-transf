# data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Import from project files ---
from config import CONFIG
from time_transf import transform_time


class XESLogLoader:
    """
    Loads event logs, builds a vocabulary from the training set,
    and uses a sentence transformer to create semantic embeddings for activities
    and resources. It also creates a stable integer mapping for activity labels.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.raw_training_logs = {}
        self.vocab = {'activity': set(), 'resource': set()}
        self.vocab_embeddings = {'activity': {}, 'resource': {}}
        self.activity_to_id = {}  # For classification labels

        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"❌ Error initializing SentenceTransformer model '{model_name}': {e}")
            print(
                "Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`) and you have an internet connection.")
            self.model = None
            self.embedding_dim = 384

        self.pad_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

    def load_and_build_vocab_from_training_logs(self, training_log_paths: dict, case_id_key='case:concept:name',
                                                activity_key='concept:name', timestamp_key='time:timestamp',
                                                resource_key='org:resource', cost_key='amount'):
        if not self.model:
            raise RuntimeError("SentenceTransformer model not available.")

        print("1. Loading training XES files to build vocabulary...")
        all_dfs = []
        for name, path in training_log_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ Warning: Training log not found at {path}. Skipping.")
                continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                df['log_name'] = name
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error reading or converting file {path}: {e}")
                continue

        if not all_dfs:
            raise ValueError("❌ No valid training logs were loaded. Cannot build vocabulary.")

        combined_df = pd.concat(all_dfs, ignore_index=True)

        print("2. Extracting unique activities/resources and creating mappings...")
        # Vocabulary for embeddings
        self.vocab['activity'].update(combined_df[activity_key].unique())
        if resource_key in combined_df.columns:
            combined_df[resource_key] = combined_df[resource_key].fillna('Unknown')
            self.vocab['resource'].update(combined_df[resource_key].unique())
        self.vocab['resource'].add('Unknown')

        # Create stable integer mapping for activity labels
        sorted_activities = sorted(list(self.vocab['activity']))
        self.activity_to_id = {name: i for i, name in enumerate(sorted_activities)}
        print(f"  - Created stable mapping for {len(self.activity_to_id)} activities.")

        print(
            f"3. Generating embeddings for {len(self.vocab['activity'])} activities and {len(self.vocab['resource'])} resources...")
        for cat_type, item_set in self.vocab.items():
            items = sorted(list(item_set))
            if items:
                embeddings = self.model.encode(items, show_progress_bar=True, normalize_embeddings=True)
                self.vocab_embeddings[cat_type] = {item: emb for item, emb in zip(items, embeddings)}

        print("4. Storing raw training log data...")
        for name, group_df in combined_df.groupby('log_name'):
            self.raw_training_logs[name] = self._convert_df_to_raw_traces(
                group_df, case_id_key, activity_key, timestamp_key, resource_key, cost_key
            )
        print("✅ Vocabulary and embeddings built successfully from training data.")

    def process_logs(self, log_paths: dict, case_id_key='case:concept:name', activity_key='concept:name',
                     timestamp_key='time:timestamp', resource_key='org:resource', cost_key='amount') -> dict:
        if not self.vocab_embeddings['activity']:
            raise RuntimeError("Vocabulary not built. Call `load_and_build_vocab_from_training_logs` first.")

        print(f"\nProcessing logs from paths: {list(log_paths.keys())}")
        processed_logs = {}
        for name, path in log_paths.items():
            if name in self.raw_training_logs:
                print(f"  - Using pre-loaded raw data for '{name}'...")
                raw_traces = self.raw_training_logs[name]
            else:
                if not os.path.exists(path):
                    print(f"⚠️ Warning: File not found at {path}. Skipping.")
                    continue
                try:
                    print(f"  - Loading and processing '{name}' from file...")
                    log = pm4py.read_xes(path)
                    df = pm4py.convert_to_dataframe(log)
                    raw_traces = self._convert_df_to_raw_traces(
                        df, case_id_key, activity_key, timestamp_key, resource_key, cost_key
                    )
                except Exception as e:
                    print(f"❌ Error processing file {path}: {e}")
                    continue

            processed_logs[name] = self._apply_embeddings_to_traces(raw_traces)

        print("✅ Log processing complete.")
        return processed_logs

    def _apply_embeddings_to_traces(self, raw_traces):
        unknown_resource_emb = self.vocab_embeddings['resource'].get('Unknown', self.pad_embedding)
        # Use a special ID for unknown activities not in the training vocab
        unknown_activity_id = -100

        log_with_embeddings = []
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                activity_emb = self.vocab_embeddings['activity'].get(event['activity'], self.pad_embedding)
                resource_emb = self.vocab_embeddings['resource'].get(event['resource'], unknown_resource_emb)
                activity_id = self.activity_to_id.get(event['activity'], unknown_activity_id)

                processed_event = {
                    'activity_embedding': activity_emb,
                    'resource_embedding': resource_emb,
                    'activity_id': activity_id,  # For classification labels
                    'cost': event['cost'],
                    'time_from_start': event['time_from_start'],
                    'time_from_previous': event['time_from_previous'],
                    'timestamp': event['timestamp'],
                    'case_id': event['case_id']
                }
                processed_trace.append(processed_event)
            log_with_embeddings.append(processed_trace)
        return log_with_embeddings

    def _convert_df_to_raw_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        raw_log = []
        df[timestamp_key] = pd.to_datetime(df[timestamp_key]).dt.tz_localize(None)
        if cost_key not in df.columns:
            print(f"⚠️ Warning: Cost attribute '{cost_key}' not found. Random costs will be generated.")

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
                    'activity': event[activity_key],
                    'timestamp': current_time.timestamp(),
                    'resource': event.get(resource_key, 'Unknown'),
                    'cost': cost_val,
                    'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time
            if trace:
                raw_log.append(trace)
        return raw_log


def get_task_data(log, task_type, max_seq_len=10):
    tasks = []
    if not log:
        return tasks

    for trace in log:
        if len(trace) < 3:
            continue

        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            if task_type == 'classification':
                label = trace[i + 1]['activity_id']
                # Ignore tasks where the label is an activity unseen during training
                if label != -100:
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
    print("--- Demonstrating Semantic Embedding Log Loader ---")

    if not os.path.isdir('./logs'):
        print("\n❌ CRITICAL: No XES files found in the './logs' directory.")
    else:
        try:
            loader = XESLogLoader()
            # 1. Build vocabulary and embeddings from training logs
            loader.load_and_build_vocab_from_training_logs(CONFIG['log_paths']['training'])

            # 2. Process a log (e.g., a test log) using the built vocabulary
            processed_logs = loader.process_logs({'D_unseen': CONFIG['log_paths']['testing']['D_unseen']})

            unseen_log_data = processed_logs.get('D_unseen')

            if not unseen_log_data:
                print("\nLog 'D_unseen' could not be processed.")
            else:
                print(f"\nSuccessfully processed log 'D_unseen' with {len(unseen_log_data)} traces.")
                flat_log = [event for trace in unseen_log_data for event in trace]
                df = pd.DataFrame(flat_log)

                print("\n--- Sample of Processed Log 'D_unseen' (first 5 events) ---")
                # Showing a subset of columns for readability
                sample_df = df[['case_id', 'activity_id', 'cost', 'time_from_start', 'time_from_previous']].head(5)
                print(sample_df.to_string())
                print(f"\nShape of activity embedding for first event: {df['activity_embedding'].iloc[0].shape}")
                print(f"Shape of resource embedding for first event: {df['resource_embedding'].iloc[0].shape}")

                print("\n--- Vocabulary Information ---")
                print(f"Total activities in vocab: {len(loader.vocab['activity'])}")
                print(f"Total resources in vocab: {len(loader.vocab['resource'])}")
                print(f"Embedding dimension: {loader.embedding_dim}")

        except (ImportError, RuntimeError) as e:
            print(f"\nEncountered an error: {e}")
