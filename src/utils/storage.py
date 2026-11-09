#  Copyright 2026 David Población Criado
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import wandb

QUANTIZED_MODEL = "quantized_model.onnx"


def split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://<bucket>/<prefix>, got '{s3_uri}'")
    return parsed.netloc, parsed.path.strip("/")


def slug(text: str) -> str:
    """W&B artifact names allow only letters, digits, '-', '_' and '.'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)


def s3_client(endpoint_url: Optional[str]):
    import boto3
    from botocore.client import Config

    # W&B resolves s3:// references through its own boto3 session, which reads
    # the endpoint only from this variable. Set it for both.
    if endpoint_url:
        os.environ["AWS_S3_ENDPOINT_URL"] = endpoint_url
    # Path-style addressing: MinIO-like endpoints do not serve
    # <bucket>.<host> virtual hosts.
    return boto3.client(
        "s3", endpoint_url=endpoint_url, config=Config(s3={"addressing_style": "path"})
    )


def _remote_size(client, bucket: str, key: str) -> Optional[int]:
    from botocore.exceptions import ClientError

    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError:
        return None


def find_quantized_models(output_dir: str) -> List[Path]:
    """Every quantized model a ramp run wrote, one per evaluated configuration"""
    return sorted(Path(output_dir, "final_results").glob(f"*/{QUANTIZED_MODEL}"))


def upload_quantized_models(
        output_dir: str,
        run: "wandb.sdk.wandb_run.Run",
        s3_uri: str,
        endpoint_url: Optional[str],
        model_name: str,
        hostname: str,
) -> Optional[str]:
    """Upload a run's quantized models and log them as a W&B reference artifact.

    Layout: ``<s3_uri>/<run id>/<config>/quantized_model.onnx``. The run id is
    unique within the project, and the model, host and config behind it are
    already in W&B, so the path does not repeat them. Files already in S3 with
    the same size are not sent again, so a failed call can simply be retried.
    The artifact is named per model and versioned across runs. Returns the S3
    prefix, or None if the run left no quantized model.
    """
    models = find_quantized_models(output_dir)
    if not models:
        logging.warning("No %s under %s/final_results, nothing to upload.", QUANTIZED_MODEL, output_dir)
        return None

    bucket, root = split_s3_uri(s3_uri)
    prefix = "/".join(p for p in (root, run.id) if p)
    client = s3_client(endpoint_url)
    total = skipped = 0
    for path in models:
        key = f"{prefix}/{path.parent.name}/{QUANTIZED_MODEL}"
        size = path.stat().st_size
        if _remote_size(client, bucket, key) == size:
            skipped += 1
            continue
        client.upload_file(str(path), bucket, key)
        total += size
    logging.info(
        "Uploaded %d quantized models (%.0f MB) to s3://%s/%s, %d already there.",
        len(models) - skipped, total / 1e6, bucket, prefix, skipped,
    )

    artifact = wandb.Artifact(
        name=f"quantized-onnx-{slug(model_name)}",
        type="model",
        description="Quantized ONNX of every evaluated configuration (stored in S3).",
        metadata={
            "host": hostname,
            "run_id": run.id,
            "s3_prefix": f"s3://{bucket}/{prefix}",
            "endpoint_url": endpoint_url,
            "configurations": [p.parent.name for p in models],
        },
    )
    # One reference for the whole prefix: W&B lists it and records every
    # object's path and ETag, not the bytes.
    artifact.add_reference(f"s3://{bucket}/{prefix}/", max_objects=len(models) + 10)
    run.log_artifact(artifact, aliases=["latest", slug(hostname)])
    return f"s3://{bucket}/{prefix}"


DEFAULT_CACHE = Path.home() / ".cache" / "ramp-mpq" / "quantized"


def run_s3_prefix(
        run_id: str, project: str = "ramp", entity: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Where a run's quantized models are in S3, and the endpoint to reach them.

    Read from the metadata of the run's ``quantized-onnx-*`` artifact, so it
    works whatever layout the upload used.
    """
    api = wandb.Api()
    run = api.run(f"{entity or api.default_entity}/{project}/{run_id}")
    for artifact in run.logged_artifacts():
        if artifact.name.startswith("quantized-onnx-"):
            return artifact.metadata["s3_prefix"], artifact.metadata.get("endpoint_url")
    raise LookupError(
        f"Run {run_id} has no quantized-onnx artifact; upload its models with "
        f"upload_quantized_models first."
    )


def download_run_models(
        run_id: str,
        configs: Optional[List[str]] = None,
        cache_dir: Path = DEFAULT_CACHE,
        project: str = "ramp",
        entity: Optional[str] = None,
) -> Dict[str, Path]:
    """Download a run's quantized models from S3; ``{config: local path}``.

    Straight from S3 with boto3 rather than ``artifact.download()``, which
    needs a writable W&B cache. Files already in ``cache_dir`` with the right
    size are reused, so comparing the same runs again downloads nothing.
    """
    s3_prefix, endpoint_url = run_s3_prefix(run_id, project, entity)
    bucket, prefix = split_s3_uri(s3_prefix)
    client = s3_client(endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL"))

    models: Dict[str, Path] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(f"/{QUANTIZED_MODEL}"):
                continue
            config = key[len(prefix) + 1:].split("/")[0]
            if configs and config not in configs:
                continue
            local = cache_dir / run_id / config / QUANTIZED_MODEL
            if not local.exists() or local.stat().st_size != obj["Size"]:
                local.parent.mkdir(parents=True, exist_ok=True)
                logging.info("Downloading %s/%s (%.0f MB)", run_id, config, obj["Size"] / 1e6)
                client.download_file(bucket, key, str(local))
            models[config] = local
    if not models:
        raise LookupError(f"No {QUANTIZED_MODEL} under {s3_prefix}")
    return dict(sorted(models.items()))
