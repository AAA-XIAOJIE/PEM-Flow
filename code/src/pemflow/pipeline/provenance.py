import numpy as np
import pandas as pd


def observation_provenance(raw, observations, audit):
    raw_keys = pd.MultiIndex.from_arrays([raw[:, 0].astype(int), raw[:, 1].astype(int)])
    keys = pd.MultiIndex.from_arrays(
        [observations[:, 0].astype(int), observations[:, 1].astype(int)],
        names=["frame", "source_track_id"],
    )
    positions = raw_keys.get_indexer(keys)
    present = positions >= 0
    confidence = np.full(len(observations), np.nan)
    confidence[present] = raw[positions[present], 7]
    provenance = pd.DataFrame(
        dict(
            observation_origin="detector_tracker",
            recovery_action="none",
            recovery_reason="",
            recovery_event_index=np.nan,
            raw_inference_key_present=present,
            raw_inference_frame=np.where(present, observations[:, 0], np.nan),
            raw_inference_source_track_id=np.where(present, observations[:, 1], np.nan),
            raw_inference_observation_id=np.where(present, positions, np.nan),
            raw_inference_confidence=confidence,
        ),
        index=keys,
    )
    for index, event in enumerate(audit["events"]):
        for action in event.get("actions", []):
            key = (int(action["frame"]), int(action["source_id"]))
            if key not in provenance.index:
                continue
            provenance.loc[key, ["recovery_action", "recovery_reason", "recovery_event_index"]] = (
                action["action"],
                action["reason"],
                index,
            )
            if action.get("replacement_observation") is not None:
                provenance.loc[key, "observation_origin"] = "bidirectional_image_optical_flow"
    return provenance.reset_index()
