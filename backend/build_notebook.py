"""Builds smart-gallery-backend.ipynb from server.py, so the notebook is one upload."""
import json
from pathlib import Path

here = Path(__file__).parent
server = (here / "server.py").read_text()


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    markdown("""
# Smart Gallery backend

Serves the fine-tuned model to the app. It stores no photos and no index: the phone keeps the vectors.

**To go live:** add secrets `NGROK_AUTH_TOKEN` and `SMART_GALLERY_API_KEY` (Add-ons → Secrets), and optionally
`NGROK_DOMAIN`. Without them the notebook runs its self-test and stops.
"""),
    code("""
!pip install -q open_clip_torch==3.3.0 transformers sentencepiece protobuf fastapi uvicorn python-multipart httpx pyngrok

import io, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import torch

print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# How long to stay up. Every hour here spends GPU quota, so idle sessions stop themselves.
MAX_HOURS = 11
IDLE_STOP_MINUTES = 60
RUN_SELF_TEST = True
SELF_TEST_IMAGES = 1000
"""),
    code("%%writefile server.py\n" + server),
    code("""
checkpoints = list(Path('/kaggle/input').rglob('smart_gallery_model.pt'))
assert checkpoints, 'Attach the output of the smart-gallery-search notebook as an input.'
checkpoint = checkpoints[0]
sys.path.insert(0, str(checkpoint.parent))
sys.path.insert(0, '.')
print('Model:', checkpoint, f'({checkpoint.stat().st_size / 1e9:.2f} GB)')

from server import Embedder, create_app, QUANT_SCALE, quantize

embedder = Embedder(checkpoint)
print(embedder.source, 'dim', embedder.dim, 'version', embedder.version)
"""),
    markdown("""
## Self-test

Runs the same path the phone uses: 384px JPEGs through the real HTTP API, compressed vectors, ranking by dot product.
"""),
    code("""
import base64
import random

from fastapi.testclient import TestClient
from PIL import Image

def find_coco():
    for root, _, files in os.walk('/kaggle/input'):
        if 'captions_val2017.json' in files:
            annotations = os.path.join(root, 'captions_val2017.json')
            break
    else:
        return None
    images = next(d for d, _, f in os.walk('/kaggle/input') if os.path.basename(d) == 'val2017')
    return annotations, images

def thumbnail(path):
    # Matches the app: a square 384px JPEG at quality 80.
    image = Image.open(path).convert('RGB').resize((384, 384), Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=80)
    return buffer.getvalue()

def decode(vector):
    return np.frombuffer(base64.b64decode(vector), dtype=np.int8)

def self_test():
    key = 'self-test-key'
    client = TestClient(create_app(embedder, key))
    headers = {'X-API-Key': key}

    health = client.get('/health').json()
    assert health['status'] == 'ok' and health['dim'] == embedder.dim, health
    assert client.post('/embed/text', json={'queries': ['a dog']}).status_code == 401
    assert client.post('/embed/text', json={'queries': ['a dog']}, headers={'X-API-Key': 'wrong'}).status_code == 401
    assert client.post('/embed/text', json={'queries': ['  ']}, headers=headers).status_code == 422
    assert client.post('/embed/text', json={'queries': []}, headers=headers).status_code == 422
    too_many = [('files', ('a.jpg', b'x', 'image/jpeg'))] * 33
    assert client.post('/embed/images', files=too_many, headers=headers).status_code == 413
    too_big = [('files', ('a.jpg', b'x' * (8 * 1024 * 1024 + 1), 'image/jpeg'))]
    assert client.post('/embed/images', files=too_big, headers=headers).status_code == 413
    print('API contract: ok')

    found = find_coco()
    assert found, 'COCO is not attached, so accuracy was not measured.'
    with open(found[0]) as f:
        data = json.load(f)
    captions = {}
    for annotation in data['annotations']:
        captions.setdefault(annotation['image_id'], []).append(annotation['caption'].strip())
    images = sorted(data['images'], key=lambda image: image['id'])
    random.Random(0).shuffle(images)
    images = images[:SELF_TEST_IMAGES]
    jpegs = [thumbnail(os.path.join(found[1], image['file_name'])) for image in images]
    print(f'Thumbnail size: {np.mean([len(j) for j in jpegs]) / 1024:.0f} KB on average')

    # One unreadable file must not lose the rest of its batch.
    mixed = [('files', ('a.jpg', jpegs[0], 'image/jpeg')), ('files', ('bad.jpg', b'not an image', 'image/jpeg'))]
    vectors = client.post('/embed/images', files=mixed, headers=headers).json()['vectors']
    assert vectors[0] is not None and vectors[1] is None, vectors
    print('Bad image handling: ok')

    started = time.time()
    compressed, exact = [], []
    for start in range(0, len(jpegs), 16):
        batch = jpegs[start:start + 16]
        files = [('files', (f'{i}.jpg', data, 'image/jpeg')) for i, data in enumerate(batch)]
        response = client.post('/embed/images', files=files, headers=headers)
        assert response.status_code == 200, response.text
        compressed += [decode(v) for v in response.json()['vectors']]
    rate = len(jpegs) / (time.time() - started)
    print(f'Image embedding: {rate:.1f} images/s')
    for start in range(0, len(jpegs), 32):
        exact += embedder.embed_image_floats(jpegs[start:start + 32])

    texts, owners = [], []
    for index, image in enumerate(images):
        texts += captions[image['id']][:5]
        owners += [index] * len(captions[image['id']][:5])
    query_compressed, query_exact = [], []
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]
        response = client.post('/embed/text', json={'queries': batch}, headers=headers)
        assert response.status_code == 200, response.text
        query_compressed += [decode(v) for v in response.json()['vectors']]
        query_exact += embedder.embed_texts([t.strip().lower() for t in batch])

    timings = []
    for i in range(20):
        started = time.time()
        client.post('/embed/text', json={'queries': [f'uncached query number {i}']}, headers=headers)
        timings.append(time.time() - started)
    print(f'Text search latency: {np.median(timings) * 1000:.0f} ms median')

    def recall(image_vectors, query_vectors):
        scores = torch.from_numpy(np.stack(query_vectors).astype(np.float32)) @ torch.from_numpy(np.stack(image_vectors).astype(np.float32)).T
        truth = torch.tensor(owners)
        ranks = (scores > scores.gather(1, truth[:, None])).sum(1).float()
        return {f'R@{k}': (ranks < k).float().mean().item() for k in (1, 5, 10)}

    exact_recall = recall(exact, query_exact)
    compressed_recall = recall(compressed, query_compressed)
    clipped = np.mean(np.abs(np.stack(compressed)) == 127)
    print('Uncompressed vectors:', exact_recall)
    print('Compressed vectors:  ', compressed_recall, f'({clipped:.4%} of values clipped)')
    assert compressed_recall['R@5'] >= exact_recall['R@5'] - 0.01, 'Compression hurts ranking too much'

    with open('self_test.json', 'w') as f:
        json.dump({
            'images': len(images), 'captions': len(texts), 'images_per_second': rate,
            'text_ms_median': float(np.median(timings) * 1000),
            'thumbnail_kb': float(np.mean([len(j) for j in jpegs]) / 1024),
            'uncompressed': exact_recall, 'compressed': compressed_recall, 'clipped': float(clipped),
        }, f, indent=2)
    print('SELF-TEST PASSED')

if RUN_SELF_TEST:
    self_test()
"""),
    markdown("""
## Serve

Uses `NGROK_DOMAIN` for a fixed address if you have one, so the app never needs its URL changed.
"""),
    code("""
import uvicorn
from kaggle_secrets import UserSecretsClient
from pyngrok import conf, ngrok

def secret(name):
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

ngrok_token, api_key, domain = secret('NGROK_AUTH_TOKEN'), secret('SMART_GALLERY_API_KEY'), secret('NGROK_DOMAIN')

if not (ngrok_token and api_key):
    print('Secrets are missing, so the server was not started. See the first cell.')
else:
    app = create_app(embedder, api_key)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=8000, log_level='warning'))
    threading.Thread(target=server.run, daemon=True).start()
    conf.get_default().auth_token = ngrok_token
    tunnel = ngrok.connect(8000, 'http', **({'domain': domain} if domain else {}))
    print('SERVER URL:', tunnel.public_url)

    started = time.time()
    app.state.last_request = started
    while True:
        time.sleep(60)
        idle = time.time() - app.state.last_request
        hours = (time.time() - started) / 3600
        print(f'{hours:.1f}h up, {app.state.embedded_images} images embedded, idle {idle / 60:.0f} min', flush=True)
        if idle > IDLE_STOP_MINUTES * 60 or hours > MAX_HOURS:
            print('Stopping to save GPU quota.')
            break
    ngrok.kill()
    server.should_exit = True
"""),
]

notebook = {
    "cells": [{**cell, "id": f"cell-{i}"} for i, cell in enumerate(cells)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(here / "kaggle" / "smart-gallery-backend.ipynb").write_text(json.dumps(notebook, indent=1))
compile(server, "server.py", "exec")
print("Wrote", here / "kaggle" / "smart-gallery-backend.ipynb")
