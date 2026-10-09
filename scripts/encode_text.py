#!/usr/bin/env python3
"""Encode item metadata JSONL into the portable custom-feature format."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from graft.io import atomic_json, sha256


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(part for child in value if (part := flatten(child)))
    if isinstance(value, dict):
        return " ".join(part for key in sorted(value) if (part := flatten(value[key])))
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-field", default="parent_asin")
    parser.add_argument("--fields", nargs="+", default=["title", "features", "description"])
    parser.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    opener = gzip.open if args.metadata.suffix == ".gz" else open
    records = {}
    with opener(args.metadata, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            item = str(record[args.id_field])
            if not item or item in records:
                raise ValueError(f"Empty/duplicate item ID on metadata line {number}")
            text = " ".join(part for field in args.fields if (part := flatten(record.get(field))))
            if not text:
                raise ValueError(f"Empty content for item {item}")
            records[item] = text
    if not records:
        raise ValueError("Metadata contains no items")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, revision=args.revision, device=args.device)
    ids = sorted(records)
    vectors = model.encode(
        [records[item] for item in ids],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise FloatingPointError("Nonfinite text embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.array(ids), vectors=vectors)
    atomic_json(
        args.output.with_suffix(".json"),
        dict(
            model=args.model,
            requested_revision=args.revision,
            resolved_revision=getattr(model[0].auto_model.config, "_commit_hash", None),
            id_field=args.id_field,
            fields=args.fields,
            items=len(ids),
            dimension=int(vectors.shape[1]),
            metadata_sha256=sha256(args.metadata),
            features_sha256=sha256(args.output),
        ),
    )


if __name__ == "__main__":
    main()
