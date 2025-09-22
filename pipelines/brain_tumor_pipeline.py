"""Kubeflow pipeline definition for automated retraining."""

from kfp import dsl
from kfp.dsl import PipelineVolume


ARTIFACT_DIR = "/app/ml/artifacts"
FEEDBACK_DIR = "/app/data/weak_feedback"


def _add_shared_volumes(op: dsl.ContainerOp, artifacts_pvc: str, feedback_pvc: str) -> dsl.ContainerOp:
    return op.add_pvolumes({
        ARTIFACT_DIR: PipelineVolume(pvc=artifacts_pvc),
        FEEDBACK_DIR: PipelineVolume(pvc=feedback_pvc),
    })


@dsl.pipeline(name="brain-tumor-retrain")
def brain_tumor_pipeline(
    image: str = "brain-tumor/api:latest",
    artifacts_pvc: str = "brain-artifacts",
    feedback_pvc: str = "brain-feedback",
    epochs: int = 3,
    batch_size: int = 16,
    metric_drop_tol: float = 0.02,
    min_auc: float = 0.7,
):
    train = dsl.ContainerOp(
        name="train-model",
        image=image,
        command=[
            "python",
            "ml/train.py",
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--out_dir",
            f"{ARTIFACT_DIR}/kfp_run",
            "--weak_dir",
            FEEDBACK_DIR,
        ],
        file_outputs={"metrics": f"{ARTIFACT_DIR}/kfp_run/metrics.json"},
    )
    train = _add_shared_volumes(train, artifacts_pvc, feedback_pvc)
    train.container.set_memory_request('12Gi')
    train.container.set_memory_limit('12Gi')
    train.container.set_cpu_request('4')
    train.container.set_cpu_limit('4')

    evaluate = dsl.ContainerOp(
        name="evaluate-metrics",
        image=image,
        command=[
            "python",
            "ml/check_metrics.py",
            "--metrics",
            f"{ARTIFACT_DIR}/kfp_run/metrics.json",
            "--state",
            f"{ARTIFACT_DIR}/state.json",
            "--min_auc",
            str(min_auc),
            "--drop_tol",
            str(metric_drop_tol),
        ],
    )
    evaluate = _add_shared_volumes(evaluate, artifacts_pvc, feedback_pvc)
    evaluate.container.set_memory_request('1Gi')
    evaluate.container.set_memory_limit('2Gi')
    evaluate.container.set_cpu_request('1')
    evaluate.after(train)

    promote = dsl.ContainerOp(
        name="promote-model",
        image=image,
        command=[
            "python",
            "ml/promote_model.py",
            "--source",
            f"{ARTIFACT_DIR}/kfp_run",
            "--target",
            f"{ARTIFACT_DIR}/current",
            "--state",
            f"{ARTIFACT_DIR}/state.json",
        ],
    )
    promote = _add_shared_volumes(promote, artifacts_pvc, feedback_pvc)
    promote.container.set_memory_request('512Mi')
    promote.container.set_memory_limit('1Gi')
    promote.container.set_cpu_request('0.5')
    promote.after(evaluate)
    
