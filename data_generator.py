# data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- Import from project files ---
from config import CONFIG
from time_transf import transform_time


class XESLogLoader:
    """
    Handles data loading and processing for training and testing.

    Workflow:
    1. `fit(training_log_paths)`: Call this first with training logs. It reads the logs
       to determine the universe of possible activities and creates a stable integer
       mapping (`activity_to_id`) for classification labels. This is a training artifact.
    2. `transform(log_paths)`: This can be called for any set of logs (training or test).
       It loads the logs, dynamically generates semantic embeddings for the activities
       and resources found within them, and uses the pre-fitted `activity_to_id` map
       to assign classification labels.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.activity_to_id = {}  # Will be populated by the `fit` method

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

    def fit(self, training_log_paths: dict, activity_key='concept:name'):
        """
        Fits the loader on the training data to define the classification label space.
        """
        if not self.model:
            raise RuntimeError("SentenceTransformer model not available.")

        print("Fitting on training data to create activity-to-ID map...")
        all_activities = set()
        for name, path in training_log_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ Warning: Training log not found at {path}. Skipping.")
                continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                all_activities.update(df[activity_key].unique())
            except Exception as e:
                print(f"❌ Error reading file {path} during fitting: {e}")
                continue

        if not all_activities:
            raise ValueError("No activities found in training logs. Cannot create mapping.")

        sorted_activities = sorted(list(all_activities))
        self.activity_to_id = {name: i for i, name in enumerate(sorted_activities)}
        print(f"✅ Activity map created with {len(self.activity_to_id)} unique activities from training data.")
        return self

    def transform(self, log_paths: dict, case_id_key='case:concept:name', activity_key='concept:name',
                  timestamp_key='time:timestamp', resource_key='org:resource', cost_key='amount'):
        """
        Loads, embeds, and processes logs using the fitted activity map.
        """
        if not self.model:
            raise RuntimeError("SentenceTransformer model not available.")
        if not self.activity_to_id:
            raise RuntimeError("Loader has not been fitted. Call `fit()` with training logs first.")

        print(f"\nTransforming logs: {list(log_paths.keys())}")

        # 1. Load all specified logs into a single DataFrame
        all_dfs = []
        for name, path in log_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ Warning: File not found at {path}. Skipping.")
                continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                df['log_name'] = name
                all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error reading file {path}: {e}")
                continue

        if not all_dfs:
            print("❌ No valid logs were loaded during transform. Returning empty dict.")
            return {}

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 2. Get unique activities/resources from THIS specific data batch for embedding
        activities_to_embed = sorted(list(combined_df[activity_key].unique()))

        resources_to_embed = set()
        if resource_key in combined_df.columns:
            combined_df[resource_key] = combined_df[resource_key].fillna('Unknown')
            resources_to_embed.update(combined_df[resource_key].unique())
        resources_to_embed.add('Unknown')
        resources_to_embed = sorted(list(resources_to_embed))

        print(
            f"  - Found {len(activities_to_embed)} unique activities and {len(resources_to_embed)} unique resources to embed.")

        # 3. Generate embeddings on the fly
        act_embeddings = self.model.encode(activities_to_embed, show_progress_bar=True, normalize_embeddings=True)
        res_embeddings = self.model.encode(resources_to_embed, show_progress_bar=True, normalize_embeddings=True)

        activity_embedding_map = {text: emb for text, emb in zip(activities_to_embed, act_embeddings)}
        resource_embedding_map = {text: emb for text, emb in zip(resources_to_embed, res_embeddings)}

        # 4. Process traces for each log
        processed_logs = {}
        for name, group_df in combined_df.groupby('log_name'):
            print(f"  - Processing traces for '{name}'...")
            raw_traces = self._convert_df_to_raw_traces(group_df, case_id_key, activity_key, timestamp_key,
                                                        resource_key, cost_key)
            processed_logs[name] = self._apply_embeddings_to_traces(raw_traces, activity_embedding_map,
                                                                    resource_embedding_map)

        print("✅ Transformation complete.")
        return processed_logs

    def _apply_embeddings_to_traces(self, raw_traces, activity_embedding_map, resource_embedding_map):
        unknown_resource_emb = resource_embedding_map.get('Unknown', self.pad_embedding)
        unknown_activity_id = -100  # Label for activities not in the training-defined map

        log_with_embeddings = []
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                # Get embeddings from the dynamically generated maps
                activity_emb = activity_embedding_map.get(event['activity'], self.pad_embedding)
                resource_emb = resource_embedding_map.get(event['resource'], unknown_resource_emb)

                # Get classification label from the pre-fitted map
                activity_id = self.activity_to_id.get(event['activity'], unknown_activity_id)

                processed_event = {
                    'activity_embedding': activity_emb,
                    'resource_embedding': resource_emb,
                    'activity_id': activity_id,
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
            # This check is now less critical as it's per-batch, but still good practice
            pass  # Suppressing warning to avoid repetition

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
                    'case_id': case_id, 'activity': event[activity_key],
                    'timestamp': current_time.timestamp(), 'resource': event.get(resource_key, 'Unknown'),
                    'cost': cost_val, 'time_from_start': (current_time - start_time).total_seconds(),
                    'time_from_previous': (current_time - prev_time).total_seconds(),
                }
                trace.append(event_dict)
                prev_time = current_time
            if trace:
                raw_log.append(trace)
        return raw_log

    def save_activity_map(self, path):
        """Saves the essential activity-to-ID map to a file."""
        map_data = {'activity_to_id': self.activity_to_id}
        torch.save(map_data, path)
        print(f"💾 Activity-to-ID map saved to {path}")

    def load_activity_map(self, path):
        """Loads the activity-to-ID map, required for stand-alone testing."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Activity map file not found at {path}. Please train a model first to generate it.")

        map_data = torch.load(path)
        self.activity_to_id = map_data['activity_to_id']
        print(f"✅ Activity-to-ID map loaded successfully from {path}")


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
                if label != -100:
                    tasks.append((prefix, label))
            elif task_type == 'regression':
                last_event_time = trace[-1]['timestamp']
                current_event_time = prefix[-1]['timestamp']
                remaining_time_hours = (last_event_time - current_event_time) / 3600.0
                label = transform_time(remaining_time_hours)
                tasks.append((prefix, label))
    return tasks
