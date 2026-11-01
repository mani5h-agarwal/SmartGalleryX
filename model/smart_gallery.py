"""Smart Gallery: natural-language photo search ("a person jumping on a mountain").

Fine-tunes a pretrained image/text model on COCO + Flickr30k and serves it through
GallerySearch. Training commands (see smart-gallery-kaggle.ipynb):

    python smart_gallery.py prepare --input /kaggle/input --out runs
    python smart_gallery.py bench   --out runs
    torchrun --standalone --nproc_per_node=gpu smart_gallery.py train --out runs
    python smart_gallery.py test    --out runs

Search only needs GallerySearch and the exported checkpoint:

    gallery = GallerySearch("smart_gallery_model.pt")
    gallery.index("/path/to/photos")
    gallery.search("a person jumping on a mountain", k=12)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import open_clip
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

CANDIDATES = [
    ("ViT-L-16-SigLIP2-384", "webli"),
    ("ViT-L-16-SigLIP-384", "webli"),
    ("ViT-H-14-quickgelu", "dfn5b"),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Flipping an image makes these captions false.
DIRECTIONAL = re.compile(r"\b(left|right)\b", re.IGNORECASE)
BLOCK = re.compile(r"(?:^|\.)(?:blocks|resblocks)\.(\d+)\.")
EMBEDDING = re.compile(r"pos_embed|positional_embedding|patch_embed|token_embedding|conv1|class_embedding")


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    return torch.autocast("cuda") if device.type == "cuda" else nullcontext()


def load_image(path):
    # Phone photos store their rotation in EXIF rather than in the pixels.
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def create_model(model_name, pretrained, device, load_weights=True):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, load_weights=load_weights, output_dict=True
    )
    return model.to(device), preprocess, open_clip.get_tokenizer(model_name)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint["model_name"], checkpoint["pretrained"]
    model, preprocess, tokenizer = create_model(*source, "cpu", load_weights=False)
    model.load_state_dict({k: v.float() for k, v in checkpoint["state_dict"].items()})
    return model.to(device).eval(), preprocess, tokenizer, source


def save_checkpoint(path, model, model_name, pretrained):
    state_dict = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model_name": model_name, "pretrained": pretrained, "state_dict": state_dict}, path)


# ---------------------------------------------------------------- data


def stable_order(keys):
    return sorted(keys, key=lambda key: hashlib.md5(str(key).encode()).hexdigest())


def read_coco(annotation_file, image_dir):
    with open(annotation_file) as f:
        data = json.load(f)
    captions = defaultdict(list)
    for annotation in data["annotations"]:
        captions[annotation["image_id"]].append(annotation["caption"].strip())
    records = []
    for image in data["images"]:
        path = os.path.join(image_dir, image["file_name"])
        if captions[image["id"]] and os.path.exists(path):
            records.append({"image": path, "captions": captions[image["id"]][:5]})
    return records


def read_flickr(results_csv, image_dirs):
    captions = defaultdict(list)
    with open(results_csv, encoding="utf-8") as f:
        next(f)
        for line in f:
            # One row of the Kaggle file has no second delimiter, so it is skipped.
            parts = [part.strip() for part in line.split("|")]
            if len(parts) == 3 and parts[2]:
                captions[parts[0]].append(parts[2])
    records = []
    for name, texts in captions.items():
        for image_dir in image_dirs:
            path = os.path.join(image_dir, name)
            if os.path.exists(path):
                records.append({"image": path, "captions": texts[:5]})
                break
    return records


def prepare(args):
    found, photo_dirs = {}, []
    for directory, _, files in os.walk(args.input):
        for name in ("captions_train2017.json", "captions_val2017.json", "results.csv"):
            if name in files:
                found.setdefault(name, os.path.join(directory, name))
        if sum(file.endswith(".jpg") for file in files) >= 100:
            photo_dirs.append(directory)
    coco_dir = {os.path.basename(path): path for path in photo_dirs}
    for name in ("train2017", "val2017"):
        if name not in coco_dir:
            raise SystemExit(f"COCO folder {name} not found under {args.input}. Attach the COCO 2017 dataset.")
    for name in ("captions_train2017.json", "captions_val2017.json"):
        if name not in found:
            raise SystemExit(f"{name} not found under {args.input}. Attach the COCO 2017 dataset.")

    coco_train = read_coco(found["captions_train2017.json"], coco_dir["train2017"])
    test_coco = read_coco(found["captions_val2017.json"], coco_dir["val2017"])
    coco_val_paths = set(stable_order(r["image"] for r in coco_train)[: args.val_coco])
    splits = {
        "train": [r for r in coco_train if r["image"] not in coco_val_paths],
        "val": [r for r in coco_train if r["image"] in coco_val_paths],
        "test_coco": test_coco,
        "test_flickr": [],
    }
    flickr_dirs = [path for path in photo_dirs if not os.path.basename(path).endswith("2017")]
    flickr = read_flickr(found["results.csv"], flickr_dirs) if "results.csv" in found else []
    if flickr:
        order = stable_order(r["image"] for r in flickr)
        test_paths, val_paths = set(order[: args.test_flickr]), set(order[args.test_flickr : args.test_flickr + args.val_flickr])
        for record in flickr:
            split = "test_flickr" if record["image"] in test_paths else "val" if record["image"] in val_paths else "train"
            splits[split].append(record)
    else:
        print("WARNING: Flickr30k not found, continuing with COCO only. Attach it for better results.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "splits.json", "w") as f:
        json.dump(splits, f)
    for name, records in splits.items():
        print(f"{name:12s} {len(records):7d} images  {sum(len(r['captions']) for r in records):7d} captions")


def read_splits(out):
    with open(Path(out) / "splits.json") as f:
        return json.load(f)


class CaptionedImages(Dataset):
    """One item per image with a random caption, so a batch never holds the same image twice."""

    def __init__(self, records, transform, tokenizer):
        self.records, self.transform, self.tokenizer = records, transform, tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        caption = random.choice(record["captions"])
        try:
            image = load_image(record["image"])
        except OSError:
            return self[random.randrange(len(self))]
        if random.random() < 0.5 and not DIRECTIONAL.search(caption):
            image = ImageOps.mirror(image)
        return self.transform(image), self.tokenizer([caption])[0]


class ImageFiles(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            return self.transform(load_image(self.paths[index])), True
        except OSError:
            return self.transform(Image.new("RGB", (32, 32))), False


def train_transform(preprocess):
    size = next(t.size for t in preprocess.transforms if hasattr(t, "size"))
    size = size if isinstance(size, int) else size[0]
    normalize = next(t for t in preprocess.transforms if isinstance(t, transforms.Normalize))
    # No hue or colour jitter: queries like "a red car" rely on true colours.
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def encode_images(model, paths, preprocess, device, batch_size, workers):
    loader = DataLoader(ImageFiles(paths, preprocess), batch_size=batch_size, num_workers=workers)
    features, valid = [], []
    for images, ok in loader:
        with autocast(device):
            output = model.encode_image(images.to(device), normalize=True)
        features.append(output.float().cpu())
        valid.append(ok)
    return torch.cat(features), torch.cat(valid)


@torch.no_grad()
def encode_texts(model, texts, tokenizer, device, batch_size):
    features = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size]).to(device)
        with autocast(device):
            output = model.encode_text(tokens, normalize=True)
        features.append(output.float().cpu())
    return torch.cat(features)


def evaluate(model, records, preprocess, tokenizer, device, batch_size=32, workers=2):
    """Text-to-image retrieval: every caption must find its own image among all images."""
    model.eval()
    images, _ = encode_images(model, [r["image"] for r in records], preprocess, device, batch_size, workers)
    captions, owners = [], []
    for index, record in enumerate(records):
        captions += record["captions"]
        owners += [index] * len(record["captions"])
    texts = encode_texts(model, captions, tokenizer, device, batch_size * 4).to(device)
    images, owners = images.to(device), torch.tensor(owners, device=device)
    ranks = []
    for start in range(0, len(texts), 1024):
        similarity = texts[start : start + 1024] @ images.T
        own = owners[start : start + 1024]
        ranks.append((similarity > similarity.gather(1, own[:, None])).sum(1))
    ranks = torch.cat(ranks).float()
    return {
        "R@1": (ranks < 1).float().mean().item(),
        "R@5": (ranks < 5).float().mean().item(),
        "R@10": (ranks < 10).float().mean().item(),
        "MRR": (1 / (ranks + 1)).mean().item(),
        "images": len(records),
        "captions": len(captions),
    }


def show(name, metrics):
    print(
        f"{name:42s} R@1 {metrics['R@1']:.4f}  R@5 {metrics['R@5']:.4f}  "
        f"R@10 {metrics['R@10']:.4f}  MRR {metrics['MRR']:.4f}  "
        f"({metrics['images']} images, {metrics['captions']} captions)",
        flush=True,
    )


def bench(args):
    device = default_device()
    records = read_splits(args.out)["val"]
    results = []
    for model_name, pretrained in CANDIDATES:
        model, preprocess, tokenizer = create_model(model_name, pretrained, device)
        metrics = evaluate(model, records, preprocess, tokenizer, device, args.batch_size, args.workers)
        show(f"{model_name} ({pretrained})", metrics)
        results.append({"model_name": model_name, "pretrained": pretrained, **metrics})
        del model
        torch.cuda.empty_cache()
    best = max(results, key=lambda result: result["MRR"])
    with open(Path(args.out) / "base_model.json", "w") as f:
        json.dump(best, f)
    print(f"Base model for fine-tuning: {best['model_name']} ({best['pretrained']})")


# ---------------------------------------------------------------- training


def retrieval_loss(image, text, logit_scale, logit_bias, world_size):
    if world_size > 1:
        images, texts = [torch.zeros_like(image) for _ in range(world_size)], [torch.zeros_like(text) for _ in range(world_size)]
        dist.all_gather(images, image.detach())
        dist.all_gather(texts, text.detach())
        rank = dist.get_rank()
        images[rank], texts[rank] = image, text
        image, text = torch.cat(images), torch.cat(texts)
    logits = logit_scale * image @ text.T
    if logit_bias is not None:
        labels = 2 * torch.eye(len(logits), device=logits.device) - 1
        loss = -F.logsigmoid(labels * (logits + logit_bias)).sum(1).mean()
    else:
        target = torch.arange(len(logits), device=logits.device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    # Each rank back-propagates through its own features only and DDP averages the
    # gradients, so scaling by world_size recovers the gradient of the global loss.
    return loss * world_size


def freeze_early_layers(model, blocks):
    """These layers get almost no learning rate anyway; freezing them frees GPU memory for a larger batch."""
    for name, parameter in model.named_parameters():
        match = BLOCK.search(name)
        if int(match[1]) < blocks if match else EMBEDDING.search(name):
            parameter.requires_grad = False


def parameter_groups(model, lr, layer_decay, weight_decay):
    """Layer-wise LR decay: the pretrained early layers move far less than the last ones."""
    depth = defaultdict(int)
    for name, _ in model.named_parameters():
        match = BLOCK.search(name)
        if match:
            tower = "image" if name.startswith("visual") else "text"
            depth[tower] = max(depth[tower], int(match[1]) + 1)

    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tower = "image" if name.startswith("visual") else "text"
        match = BLOCK.search(name)
        if match:
            layer = int(match[1]) + 1
        elif EMBEDDING.search(name):
            layer = 0
        else:
            layer = depth[tower] + 1
        scale = layer_decay ** (depth[tower] + 1 - layer)
        no_decay = parameter.ndim < 2 or "logit_" in name
        key = (scale, no_decay)
        group = groups.setdefault(
            key, {"params": [], "base_lr": lr * scale, "lr": lr * scale, "weight_decay": 0.0 if no_decay else weight_decay}
        )
        group["params"].append(parameter)
    return list(groups.values())


def fit_batch_size(model, optimizer, tokenizer, image_size, device, batch_size):
    """Largest batch that survives a full step, including the optimizer's own memory."""
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    model.train()
    while batch_size >= 4:
        try:
            images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
            tokens = tokenizer(["a photo"] * batch_size).to(device)
            # Two steps: the second forward pass runs with the optimizer state allocated.
            for _ in range(2):
                with autocast(device):
                    output = model(images, tokens)
                    loss = (output["image_features"] * output["text_features"]).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            break
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size * 3 // 4
        finally:
            images = tokens = output = loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    else:
        raise SystemExit("Model does not fit on this GPU.")
    optimizer.state.clear()
    for group, lr in zip(optimizer.param_groups, learning_rates):
        group["lr"] = lr
    # Real batches fragment memory more than this probe does.
    return max(4, batch_size * 9 // 10)


def train(args):
    distributed = "RANK" in os.environ
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=timedelta(hours=3))
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = default_device()
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    elif distributed:
        device = torch.device("cpu")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    is_main = rank == 0
    log = print if is_main else (lambda *_, **__: None)

    out = Path(args.out)
    splits = read_splits(out)
    model_name, pretrained = args.model, args.pretrained
    if not model_name:
        with open(out / "base_model.json") as f:
            base = json.load(f)
        model_name, pretrained = base["model_name"], base["pretrained"]

    model, preprocess, tokenizer = create_model(model_name, pretrained, device)
    freeze_early_layers(model, args.freeze_blocks)
    model.set_grad_checkpointing(True)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr, args.layer_decay, args.weight_decay), betas=(0.9, 0.98))
    transform = train_transform(preprocess)
    batch_size = args.batch_size
    if device.type == "cuda":
        image_size = transform.transforms[0].size[0]
        batch_size = fit_batch_size(model, optimizer, tokenizer, image_size, device, batch_size)
        if distributed:
            agreed = torch.tensor(batch_size, device=device)
            dist.all_reduce(agreed, dist.ReduceOp.MIN)
            batch_size = int(agreed)

    dataset = CaptionedImages(splits["train"], transform, tokenizer)
    sampler = DistributedSampler(dataset, drop_last=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    total_steps = args.max_steps or args.epochs * len(loader)
    log(f"{model_name} ({pretrained}): {len(dataset)} images, {batch_size} x {world_size} per step, {total_steps} steps", flush=True)

    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    if is_main:
        metrics = evaluate(raw_model, splits["val"], preprocess, tokenizer, device, batch_size * 2, args.workers)
        show("pretrained, validation", metrics)
        best = metrics["MRR"]
        # The pretrained weights are the bar to beat: fine-tuning can only replace them on merit.
        save_checkpoint(out / "best.pt", raw_model, model_name, pretrained)
    if distributed:
        dist.barrier()

    step, epoch, started, timed_from = 0, 0, time.time(), None
    eval_every = args.eval_every or max(1, total_steps // 6)
    model.train()
    while step < total_steps:
        if sampler:
            sampler.set_epoch(epoch)
        for images, tokens in loader:
            if step >= total_steps:
                break
            warmup = min(1.0, (step + 1) / args.warmup_steps)
            progress = step / max(1, total_steps)
            factor = warmup * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor

            with autocast(device):
                output = model(images.to(device, non_blocking=True), tokens.to(device, non_blocking=True))
            loss = retrieval_loss(
                output["image_features"].float(), output["text_features"].float(),
                output["logit_scale"].float(), output.get("logit_bias"), world_size,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 20:
                timed_from = time.time()
            if step == 120:
                # Shrink the schedule, not just stop early, so the LR still decays to zero.
                seconds_per_step = (time.time() - timed_from) / 100
                remaining = args.max_hours * 3600 * 0.85 - (time.time() - started)
                limit = torch.tensor(step + max(0, int(remaining / seconds_per_step)), device=device)
                if distributed:
                    dist.broadcast(limit, 0)
                if int(limit) < total_steps:
                    total_steps, eval_every = int(limit), max(1, int(limit) // 6)
                log(f"{seconds_per_step:.2f}s per step, running {total_steps} steps", flush=True)
            if step % 50 == 0:
                elapsed = (time.time() - started) / 3600
                log(f"step {step}/{total_steps} loss {loss.item() / world_size:.3f} lr {optimizer.param_groups[-1]['lr']:.2e} {elapsed:.2f}h", flush=True)

            if step % eval_every == 0 or step == total_steps:
                if is_main:
                    metrics = evaluate(raw_model, splits["val"], preprocess, tokenizer, device, batch_size * 2, args.workers)
                    show(f"step {step}, validation", metrics)
                    if metrics["MRR"] > best:
                        best = metrics["MRR"]
                        save_checkpoint(out / "best.pt", raw_model, model_name, pretrained)
                        print("  new best, saved", flush=True)
                if distributed:
                    dist.barrier()
                model.train()
        epoch += 1
    if distributed:
        dist.destroy_process_group()


def test(args):
    """Scores on data untouched by training and model selection: base model vs final model."""
    device = default_device()
    splits = read_splits(args.out)
    out = Path(args.out)
    tuned, preprocess, tokenizer, source = load_checkpoint(out / "best.pt", device)
    base, _, _ = create_model(*source, device)
    results = {}
    for split in ("test_coco", "test_flickr"):
        if not splits[split]:
            continue
        for label, model in (("pretrained", base), ("final", tuned)):
            results[f"{split}/{label}"] = evaluate(model, splits[split], preprocess, tokenizer, device, args.batch_size, args.workers)
            show(f"{split} {label}", results[f"{split}/{label}"])
    with open(out / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------- search


class GallerySearch:
    def __init__(self, checkpoint, device=None):
        self.device = torch.device(device) if device else default_device()
        self.model, self.preprocess, self.tokenizer, _ = load_checkpoint(checkpoint, self.device)
        self.paths, self.features = [], torch.empty(0)

    def index(self, folder, cache=None, batch_size=16, workers=2):
        """Embeds every photo under folder. With cache, only new or changed photos are embedded."""
        files = sorted(
            str(path) for path in Path(folder).rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
        )
        stamps = {path: os.path.getmtime(path) for path in files}
        known = {}
        if cache and os.path.exists(cache):
            saved = np.load(cache, allow_pickle=False)
            for path, stamp, feature in zip(saved["paths"].tolist(), saved["stamps"], saved["features"]):
                if stamps.get(path) == stamp:
                    known[path] = feature
        todo = [path for path in files if path not in known]
        if todo:
            features, valid = encode_images(self.model, todo, self.preprocess, self.device, batch_size, workers)
            for path, feature, ok in zip(todo, features.numpy(), valid.tolist()):
                if ok:
                    known[path] = feature
        self.paths = [path for path in files if path in known]
        if not self.paths:
            raise ValueError(f"No readable photos under {folder}")
        self.features = torch.from_numpy(np.stack([known[path] for path in self.paths]))
        if cache:
            np.savez(
                cache, paths=np.array(self.paths), features=self.features.numpy(),
                stamps=np.array([stamps[path] for path in self.paths]),
            )
        print(f"Indexed {len(self.paths)} photos ({len(todo)} new)")

    def search(self, query, k=12):
        text = encode_texts(self.model, [query], self.tokenizer, self.device, 1)[0]
        scores = self.features @ text
        top = scores.topk(min(k, len(scores)))
        return [(self.paths[i], score) for i, score in zip(top.indices.tolist(), top.values.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("prepare")
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="runs")
    p.add_argument("--val-coco", type=int, default=2000)
    p.add_argument("--val-flickr", type=int, default=1000)
    p.add_argument("--test-flickr", type=int, default=1000)
    p.set_defaults(run=prepare)

    p = commands.add_parser("bench")
    p.add_argument("--out", default="runs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(run=bench)

    p = commands.add_parser("train")
    p.add_argument("--out", default="runs")
    p.add_argument("--model", help="overrides the bench winner, for example ViT-L-16-SigLIP2-384")
    p.add_argument("--pretrained", default="webli")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=9.5)
    p.add_argument("--batch-size", type=int, default=48, help="per GPU; lowered automatically if it does not fit")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--layer-decay", type=float, default=0.85)
    p.add_argument("--freeze-blocks", type=int, default=8, help="blocks frozen at the start of each tower")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--workers", type=int, default=2, help="per GPU")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run=train)

    p = commands.add_parser("test")
    p.add_argument("--out", default="runs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(run=test)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
