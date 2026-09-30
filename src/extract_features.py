from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PRESETS = {
    "PACS": {
        "classes": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
        "domains": ["art_painting", "cartoon", "photo", "sketch"],
    },
    "OfficeHome": {
        "classes": [
            "Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket",
            "Calculator", "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch",
            "Curtains", "Desk_Lamp", "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet",
            "Flipflops", "Flowers", "Folder", "Fork", "Glasses", "Hammer", "Helmet", "Kettle",
            "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker", "Monitor", "Mop", "Mouse",
            "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil", "Postit_Notes",
            "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors", "Screwdriver",
            "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "TV", "Table",
            "Telephone", "ToothBrush", "Toys", "Trash_Can", "Webcam",
        ],
        "domains": ["Art", "Clipart", "Product", "Real World"],
    },
    "TerraIncognita": {
        "classes": [
            "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
            "raccoon", "squirrel",
        ],
        "domains": ["location_38", "location_43", "location_46", "location_100"],
    },
}


class FolderBenchmark(Dataset):
    def __init__(self, root: Path, classes: list[str], domains: list[str], preprocess: Callable):
        self.root = root
        self.preprocess = preprocess
        self.samples: list[tuple[Path, int, int]] = []
        for domain_id, domain in enumerate(domains):
            for label, class_name in enumerate(classes):
                folder = root / domain / class_name
                if not folder.is_dir():
                    raise FileNotFoundError(folder)
                for path in sorted(folder.iterdir()):
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        self.samples.append((path, label, domain_id))
        if not self.samples:
            raise RuntimeError(f"No benchmark images under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, domain = self.samples[index]
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, label, domain, str(path.relative_to(self.root))


class OfficeHomeParquet(Dataset):
    """Read the public flwrlabs/office-home parquet snapshot without relabeling."""

    def __init__(self, root: Path, classes: list[str], domains: list[str], preprocess: Callable):
        import pandas as pd

        files = sorted(root.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet shards under {root}")
        frame = pd.concat(
            [pd.read_parquet(path, columns=["image", "domain", "label"]) for path in files],
            ignore_index=True,
        )
        domain_to_id = {name: index for index, name in enumerate(domains)}
        unknown = set(frame["domain"].unique()) - set(domain_to_id)
        if unknown:
            raise ValueError(f"Unknown Office-Home domains: {sorted(unknown)}")
        if int(frame["label"].min()) != 0 or int(frame["label"].max()) != len(classes) - 1:
            raise ValueError("Office-Home label range does not match the frozen class list")
        self.rows = frame.to_dict("records")
        self.domain_to_id = domain_to_id
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_value = row["image"]
        image_bytes = image_value["bytes"] if isinstance(image_value, dict) else image_value
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor = self.preprocess(image.convert("RGB"))
        domain = self.domain_to_id[str(row["domain"])]
        return tensor, int(row["label"]), domain, f"parquet/{index:05d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(args, device: str):
    if args.backend == "openai":
        import sys

        if args.clip_source is None:
            raise ValueError("--clip-source is required for the OpenAI backend")
        sys.path.insert(0, str(args.clip_source))
        import clip  # type: ignore

        model, preprocess = clip.load(str(args.weights), device=device, jit=False)
        return model, preprocess, clip.tokenize, "OpenAI CLIP"

    import open_clip  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=str(args.weights), device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    return model, preprocess, tokenizer, "OpenCLIP"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-format", choices=["folder", "officehome_parquet"], default="folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai", "open_clip"], default="openai")
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--clip-source", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    preset = PRESETS[args.dataset]
    classes = list(preset["classes"])
    domains = list(preset["domains"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenize, family = load_model(args, device)
    model.eval()
    if args.data_format == "officehome_parquet":
        dataset: Dataset = OfficeHomeParquet(args.data_root, classes, domains, preprocess)
    else:
        dataset = FolderBenchmark(args.data_root, classes, domains, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )

    features, labels, domain_ids, paths = [], [], [], []
    with torch.inference_mode():
        for images, batch_labels, batch_domains, batch_paths in loader:
            encoded = model.encode_image(images.to(device, non_blocking=True)).float()
            encoded /= encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.cpu().numpy().astype(np.float32))
            labels.append(batch_labels.numpy().astype(np.int64))
            domain_ids.append(batch_domains.numpy().astype(np.int64))
            paths.extend(batch_paths)

        prompt_classes = [name.replace("_", " ").lower() for name in classes]
        prompts = [f"a photo of a {name}." for name in prompt_classes]
        tokens = tokenize(prompts).to(device)
        text = model.encode_text(tokens).float()
        text /= text.norm(dim=-1, keepdim=True)
        text_features = text.cpu().numpy().astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        domains=np.concatenate(domain_ids),
        paths=np.asarray(paths),
        text_features=text_features,
        classes=np.asarray(classes),
        domain_names=np.asarray(domains),
    )
    metadata = {
        "dataset": args.dataset,
        "samples": len(dataset),
        "classes": classes,
        "domains": domains,
        "backbone": f"{family} {args.model_name}",
        "backend": args.backend,
        "data_format": args.data_format,
        "prompt": "a photo of a {class}.",
        "weights_sha256": sha256(args.weights),
        "cache_sha256": sha256(args.output),
        "torch": torch.__version__,
        "device": device,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
