# data_generator.py
import pandas as pd
import random
import pm4py
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# --- Import from project files ---
from config import CONFIG
from time_transf import transform_time


class XESLogLoader:
    """
    Handles data loading and processing. It learns an activity map from training
    data and can map unseen test activities to their closest semantic match
    using the Hungarian algorithm for optimal 1-to-1 assignment.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Training-derived artifacts
        self.activity_to_id = {}
        self.training_activity_names = []
        self.training_activity_embeddings = None

        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"❌ Error initializing SentenceTransformer model '{model_name}': {e}")
            self.model = None
            self.embedding_dim = 384

        self.pad_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

    def fit(self, training_log_paths: dict, activity_key='concept:name'):
        """
        Fits on training data to create the activity->ID map and store
        the embeddings for all known training activities.
        """
        if not self.model: raise RuntimeError("SentenceTransformer model not available.")

        print("Fitting on training data to create activity map and embeddings...")
        all_activities = set()
        for _, path in training_log_paths.items():
            if not os.path.exists(path): continue
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                all_activities.update(df[activity_key].unique())
            except Exception as e:
                print(f"❌ Error reading file {path} during fitting: {e}")

        if not all_activities: raise ValueError("No activities found in training logs.")

        self.training_activity_names = sorted(list(all_activities))
        self.activity_to_id = {name: i for i, name in enumerate(self.training_activity_names)}

        print(f"  - Created map for {len(self.activity_to_id)} unique activities.")
        print("  - Generating and storing embeddings for all training activities...")
        self.training_activity_embeddings = self.model.encode(
            self.training_activity_names, show_progress_bar=True, normalize_embeddings=True
        )
        print("✅ Fit complete.")
        return self

    def transform(self, log_paths: dict, case_id_key='case:concept:name', activity_key='concept:name',
                  timestamp_key='time:timestamp', resource_key='org:resource', cost_key='amount'):
        """
        Loads and processes logs. Unseen activities are mapped to the most
        semantically similar training activity via an optimal 1-to-1 assignment.
        """
        if not self.model: raise RuntimeError("SentenceTransformer model not available.")
        if not self.activity_to_id: raise RuntimeError("Loader has not been fitted.")

        print(f"\nTransforming logs: {list(log_paths.keys())}")

        all_dfs = [pm4py.convert_to_dataframe(pm4py.read_xes(path)) for path in log_paths.values() if
                   os.path.exists(path)]
        if not all_dfs: return {}
        combined_df = pd.concat(all_dfs, keys=log_paths.keys(), names=['log_name', 'orig_index']).reset_index()

        current_activities = sorted(list(combined_df[activity_key].unique()))

        # --- Create the complete activity->ID map for this batch ---
        final_activity_id_map = self.activity_to_id.copy()
        unseen_activities = [name for name in current_activities if name not in self.activity_to_id]

        if unseen_activities:
            print(f"  - Found {len(unseen_activities)} unseen activities. Mapping them to training vocabulary...")
            unseen_embeddings = self.model.encode(unseen_activities, normalize_embeddings=True)

            # Calculate cosine similarity and create a cost matrix for the assignment problem
            similarity_matrix = cosine_similarity(unseen_embeddings, self.training_activity_embeddings)
            cost_matrix = 1 - similarity_matrix  # Hungarian algorithm minimizes cost, so we use 1 - similarity

            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            print(f"  - Optimal 1-to-1 mapping found for {len(row_ind)} activities.")
            for r, c in zip(row_ind, col_ind):
                unseen_name = unseen_activities[r]
                mapped_name = self.training_activity_names[c]
                mapped_id = self.activity_to_id[mapped_name]
                final_activity_id_map[unseen_name] = mapped_id
                # print(f"    '{unseen_name}' -> '{mapped_name}' (ID: {mapped_id})") # Uncomment for debugging

        # --- Embed all activities and resources for this batch ---
        resources_to_embed = sorted(list(combined_df[resource_key].fillna('Unknown').unique()))

        activity_embedding_map = {name: emb for name, emb in zip(current_activities,
                                                                 self.model.encode(current_activities,
                                                                                   normalize_embeddings=True))}
        resource_embedding_map = {name: emb for name, emb in zip(resources_to_embed,
                                                                 self.model.encode(resources_to_embed,
                                                                                   normalize_embeddings=True))}

        # --- Process traces using the final maps ---
        processed_logs = {}
        for name, group_df in combined_df.groupby('log_name'):
            raw_traces = self._convert_df_to_raw_traces(group_df, case_id_key, activity_key, timestamp_key,
                                                        resource_key, cost_key)
            processed_logs[name] = self._apply_embeddings_to_traces(raw_traces, activity_embedding_map,
                                                                    resource_embedding_map, final_activity_id_map)

        print("✅ Transformation complete.")
        return processed_logs

    def _apply_embeddings_to_traces(self, raw_traces, activity_embedding_map, resource_embedding_map,
                                    final_activity_id_map):
        unknown_resource_emb = resource_embedding_map.get('Unknown', self.pad_embedding)

        log_with_embeddings = []
        for raw_trace in raw_traces:
            processed_trace = []
            for event in raw_trace:
                activity_id = final_activity_id_map.get(event['activity'], -100)

                processed_event = {
                    'activity_embedding': activity_embedding_map.get(event['activity'], self.pad_embedding),
                    'resource_embedding': resource_embedding_map.get(event['resource'], unknown_resource_emb),
                    'activity_id': activity_id,
                    'cost': event['cost'],
                    'time_from_start': event['time_from_start'],
                    'time_from_previous': event['time_from_previous'],
                    'timestamp': event['timestamp'], 'case_id': event['case_id']
                }
                processed_trace.append(processed_event)
            log_with_embeddings.append(processed_trace)
        return log_with_embeddings

    def _convert_df_to_raw_traces(self, df, case_id_key, activity_key, timestamp_key, resource_key, cost_key):
        # This function remains the same as before
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

    def save_training_artifacts(self, path):
        """Saves the essential training artifacts needed for testing."""
        artifacts = {
            'activity_to_id': self.activity_to_id,
            'training_activity_names': self.training_activity_names,
            'training_activity_embeddings': self.training_activity_embeddings
        }
        torch.save(artifacts, path)
        print(f"💾 Training artifacts saved to {path}")

    def load_training_artifacts(self, path):
        """Loads artifacts, required for stand-alone testing."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifacts file not found at {path}.")

        artifacts = torch.load(path)
        self.activity_to_id = artifacts['activity_to_id']
        self.training_activity_names = artifacts['training_activity_names']
        self.training_activity_embeddings = artifacts['training_activity_embeddings']
        print(f"✅ Training artifacts loaded successfully from {path}")


def get_task_data(log, task_type, max_seq_len=10):
    # This function is now slightly simpler as the label is always valid
    tasks = []
    if not log: return tasks

    for trace in log:
        if len(trace) < 3: continue
        for i in range(1, len(trace) - 1):
            prefix = trace[:i + 1]
            if len(prefix) > max_seq_len:
                prefix = prefix[-max_seq_len:]

            if task_type == 'classification':
                # The label should now always be valid due to the mapping
                label = trace[i + 1]['activity_id']
                if label != -100:  # Keep safeguard just in case
                    tasks.append((prefix, label))
            elif task_type == 'regression':
                last_event_time = trace[-1]['timestamp']
                current_event_time = prefix[-1]['timestamp']
                remaining_time_hours = (last_event_time - current_event_time) / 3600.0
                label = transform_time(remaining_time_hours)
                tasks.append((prefix, label))
    return tasks
