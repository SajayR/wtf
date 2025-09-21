#!/usr/bin/env python3
import os, time, json
from pathlib import Path
import kfp
from ml.retrain import count_feedback, load_state, save_state  # reuse your helpers

def main() -> int:
    # Where state + feedback live (already on PVCs in your manifests)
    artifacts_dir = Path(os.getenv("ARTIFACT_DIR", "ml/artifacts"))
    state_path = artifacts_dir / "state.json"
    weak_dir = Path(os.getenv("WEAK_FEEDBACK_DIR", "data/weak_feedback"))
    threshold = int(os.getenv("RETRAIN_THRESHOLD", "5"))

    # KFP connection + pipeline package
    host = os.getenv("KFP_HOST")  # e.g. http://ml-pipeline.kubeflow.svc.cluster.local:8888
    if not host:
        print("[kfp-submit] KFP_HOST not set"); return 1
    spec_path = Path(os.getenv("PIPELINE_SPEC", "pipelines/brain_tumor_pipeline.yaml"))
    experiment = os.getenv("KFP_EXPERIMENT", "brain-tumor")
    namespace = os.getenv("KFP_NAMESPACE")  # optional
    image = os.getenv("PIPELINE_IMAGE", "brain-tumor/api:latest")

    # Count new feedback since last time
    state = load_state(state_path)
    seen = int(state.get("feedback_seen", 0))
    current = count_feedback(weak_dir)
    new = max(0, current - seen)
    print(f"[kfp-submit] feedback_seen={seen} current={current} new={new} threshold={threshold}")
    if new < threshold:
        print("[kfp-submit] below threshold, not submitting")
        return 0

    # Submit a KFP run
    client_kwargs = {"host": host}
    if namespace:
        client_kwargs["namespace"] = namespace
    client = kfp.Client(**client_kwargs)

    run_name = f"retrain-{int(time.time())}"
    print(f"[kfp-submit] submitting {run_name} with image={image}")
    result = client.create_run_from_pipeline_package(
        str(spec_path),
        arguments={
            "image": image,
            "artifacts_pvc": os.getenv("ARTIFACTS_PVC", "brain-artifacts"),
            "feedback_pvc": os.getenv("FEEDBACK_PVC", "brain-feedback"),
            "epochs": int(os.getenv("EPOCHS", "3")),
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
            "metric_drop_tol": float(os.getenv("METRIC_DROP_TOL", "0.02")),
            "min_auc": float(os.getenv("MIN_AUC", "0.7")),
        },
        run_name=run_name,
        experiment_name=experiment,
    )
    print(f"[kfp-submit] run_id={result.run_id}")

    # Mark these images as “seen” to avoid duplicate submits
    state["feedback_seen"] = current
    save_state(state_path, state)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
