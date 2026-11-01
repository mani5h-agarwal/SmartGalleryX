"""Smart Gallery backend: a stateless image and text embedding service.

The server keeps no photos and no index. The app sends small thumbnails, stores the
vectors it gets back, and ranks them on the phone, so a Kaggle restart loses nothing.

    GET  /health        model details, no key needed
    POST /embed/images  multipart "files", up to MAX_FILES images
    POST /embed/text    {"queries": ["a dog in the snow"]}

Vectors travel as base64 int8 (QUANT_SCALE * value, clipped to +-127): 1 KB per photo
instead of 4 KB, for a loss in ranking measured by the notebook's self-test.
"""
import base64
import hashlib
import hmac
import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageFile, ImageOps
from pydantic import BaseModel, Field

from smart_gallery import load_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_FILES = 32
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_QUERIES = 16
MAX_QUERY_CHARS = 300
QUANT_SCALE = 512.0
TEXT_CACHE_SIZE = 4096


def quantize(vectors):
    return np.clip(np.rint(vectors * QUANT_SCALE), -127, 127).astype(np.int8)


def encode_vector(vector):
    return base64.b64encode(vector.tobytes()).decode("ascii")


def model_fingerprint(path):
    """Identifies the weights. The app rebuilds its index when this changes."""
    path = Path(path)
    digest = hashlib.sha256(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(1024 * 1024))
    return digest.hexdigest()[:12]


class Embedder:
    """The image tower runs on the first GPU and the text tower on the second, so
    searches stay fast while a phone is indexing."""

    def __init__(self, checkpoint, decode_workers=4):
        count = torch.cuda.device_count()
        self.image_device = torch.device("cuda:0" if count else "cpu")
        self.text_device = torch.device(f"cuda:{count - 1}" if count else "cpu")
        self.model, self.preprocess, self.tokenizer, self.source = load_checkpoint(checkpoint, self.image_device)
        self.model.text.to(self.text_device)
        self.version = model_fingerprint(checkpoint)
        self.image_lock = threading.Lock()
        self.text_lock = threading.Lock() if count > 1 else self.image_lock
        self.decoders = ThreadPoolExecutor(decode_workers)
        self.text_cache = OrderedDict()
        self.dim = len(self.embed_texts(["a photo"])[0])

    def _autocast(self, device):
        return torch.autocast("cuda", enabled=device.type == "cuda")

    def _decode(self, data):
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            return self.preprocess(image)
        except Exception:
            return None

    def embed_image_floats(self, images):
        """Returns one vector per input, or None for an image that could not be read."""
        tensors = list(self.decoders.map(self._decode, images))
        usable = [i for i, tensor in enumerate(tensors) if tensor is not None]
        results = [None] * len(images)
        if not usable:
            return results
        batch = torch.stack([tensors[i] for i in usable])
        with self.image_lock, torch.inference_mode(), self._autocast(self.image_device):
            vectors = self.model.encode_image(batch.to(self.image_device), normalize=True)
        vectors = vectors.float().cpu().numpy()
        for index, vector in zip(usable, vectors):
            results[index] = vector
        return results

    def embed_texts(self, queries):
        with self.text_lock:
            found = {}
            missing = []
            for query in dict.fromkeys(queries):
                if query in self.text_cache:
                    self.text_cache.move_to_end(query)
                    found[query] = self.text_cache[query]
                else:
                    missing.append(query)
            if missing:
                tokens = self.tokenizer(missing).to(self.text_device)
                with torch.inference_mode(), self._autocast(self.text_device):
                    vectors = self.model.encode_text(tokens, normalize=True)
                for query, vector in zip(missing, vectors.float().cpu().numpy()):
                    found[query] = vector
                    self.text_cache[query] = vector
                while len(self.text_cache) > TEXT_CACHE_SIZE:
                    self.text_cache.popitem(last=False)
        return [found[query] for query in queries]


class TextRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=MAX_QUERIES)


def create_app(embedder, api_key):
    if not api_key:
        raise ValueError("An API key is required: the server is reachable from the internet.")

    app = FastAPI(title="Smart Gallery", docs_url=None, redoc_url=None)
    app.state.last_request = time.time()
    app.state.embedded_images = 0

    def require_key(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(401, "Invalid API key")
        app.state.last_request = time.time()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": embedder.source[0],
            "model_version": embedder.version,
            "dim": embedder.dim,
            "quant_scale": QUANT_SCALE,
            "max_files": MAX_FILES,
            "gpus": torch.cuda.device_count(),
        }

    @app.post("/embed/images", dependencies=[Depends(require_key)])
    async def embed_images(files: list[UploadFile] = File()):
        if len(files) > MAX_FILES:
            raise HTTPException(413, f"Send at most {MAX_FILES} images at once")
        images = []
        for file in files:
            data = await file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{file.filename} is too large; send a thumbnail")
            images.append(data)
        vectors = await run_in_threadpool(embedder.embed_image_floats, images)
        app.state.embedded_images += sum(vector is not None for vector in vectors)
        return {
            "model_version": embedder.version,
            "vectors": [None if v is None else encode_vector(quantize(v)) for v in vectors],
        }

    @app.post("/embed/text", dependencies=[Depends(require_key)])
    async def embed_text(request: TextRequest):
        queries = [query.strip().lower()[:MAX_QUERY_CHARS] for query in request.queries]
        if not all(queries):
            raise HTTPException(422, "Queries must not be empty")
        vectors = await run_in_threadpool(embedder.embed_texts, queries)
        return {
            "model_version": embedder.version,
            "vectors": [encode_vector(quantize(vector)) for vector in vectors],
        }

    return app
