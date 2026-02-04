#!/usr/bin/env python3
from pm4py.util import xes_constants


def _collect_activities_and_paths(log, activity_key):
    activities = set()
    paths = set()
    for trace in log:
        events = [event for event in trace if activity_key in event]
        for event in events:
            activities.add(event[activity_key])
        for idx in range(1, len(events)):
            paths.add((events[idx - 1][activity_key], events[idx][activity_key]))
    activities = sorted(activities)
    paths = sorted(paths)
    return activities, paths


def _build_prefix_features(log, activity_key, timestamp_key, target_mode):
    activities, paths = _collect_activities_and_paths(log, activity_key)
    activity_to_index = {activity: idx for idx, activity in enumerate(activities)}
    path_to_index = {path: idx for idx, path in enumerate(paths)}

    feature = []
    target = []
    case_ids = []

    for trace_index, trace in enumerate(log):
        events = [event for event in trace if activity_key in event]
        if len(events) < 2:
            continue
        trace_id = trace.attributes.get(
            xes_constants.DEFAULT_TRACEID_KEY, trace_index
        )
        end_time = events[-1].get(timestamp_key) if timestamp_key else None
        if target_mode == "remaining_time" and end_time is None:
            continue
        start_time = events[0].get(timestamp_key) if timestamp_key else None

        seen_activities = {events[0][activity_key]}
        seen_paths = set()

        for idx in range(1, len(events)):
            prev_event = events[idx - 1]
            curr_event = events[idx]

            if idx >= 2:
                prior_event = events[idx - 2]
                seen_paths.add(
                    (prior_event[activity_key], prev_event[activity_key])
                )

            row = [0] * (len(activities) + len(paths))
            for seen_activity in seen_activities:
                row[activity_to_index[seen_activity]] = 1
            for seen_path in seen_paths:
                row[len(activities) + path_to_index[seen_path]] = 1

            prev_to_penultimate = 0.0
            if idx >= 2 and timestamp_key:
                prev_ts = prev_event.get(timestamp_key)
                prior_ts = events[idx - 2].get(timestamp_key)
                if prev_ts is not None and prior_ts is not None:
                    prev_to_penultimate = (prev_ts - prior_ts).total_seconds()

            start_to_penultimate = 0.0
            if timestamp_key and start_time is not None:
                penultimate_ts = prev_event.get(timestamp_key)
                if penultimate_ts is not None:
                    start_to_penultimate = (penultimate_ts - start_time).total_seconds()

            path_time_diff = 0.0
            if timestamp_key:
                prev_ts = prev_event.get(timestamp_key)
                curr_ts = curr_event.get(timestamp_key)
                if prev_ts is not None and curr_ts is not None:
                    path_time_diff = (curr_ts - prev_ts).total_seconds()

            row.extend([prev_to_penultimate, start_to_penultimate, path_time_diff])

            curr_activity = curr_event[activity_key]
            if target_mode == "next_activity":
                target.append(activity_to_index[curr_activity])
                feature.append(row)
            else:
                curr_ts = curr_event.get(timestamp_key) if timestamp_key else None
                if curr_ts is not None:
                    remaining = (end_time - curr_ts).total_seconds()
                    target.append(remaining)
                    feature.append(row)
                    case_ids.append(trace_id)

            seen_activities.add(curr_activity)

    return feature, target, case_ids, activities, activity_to_index, paths, path_to_index


def build_prefix_features_next_activity(
    log,
    activity_key=xes_constants.DEFAULT_NAME_KEY,
    timestamp_key=xes_constants.DEFAULT_TIMESTAMP_KEY,
):
    feature, target, _, activities, activity_to_index, paths, path_to_index = (
        _build_prefix_features(log, activity_key, timestamp_key, "next_activity")
    )
    return feature, target, activities, activity_to_index, paths, path_to_index


def build_prefix_features_remaining_time(
    log,
    activity_key=xes_constants.DEFAULT_NAME_KEY,
    timestamp_key=xes_constants.DEFAULT_TIMESTAMP_KEY,
):
    feature, target, case_ids, activities, activity_to_index, paths, path_to_index = (
        _build_prefix_features(log, activity_key, timestamp_key, "remaining_time")
    )
    return feature, target, case_ids, activities, activity_to_index, paths, path_to_index
