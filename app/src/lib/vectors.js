// In-memory photo vectors. Everything lives in one Int8Array so ranking a whole
// library touches no JS objects, and never blocks the UI for more than a frame or two.

const YIELD_EVERY = 2048;

const nextTick = () => new Promise((resolve) => setTimeout(resolve, 0));

export function decodeVector(base64) {
  const binary = atob(base64);
  const vector = new Int8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    const byte = binary.charCodeAt(i);
    vector[i] = byte > 127 ? byte - 256 : byte;
  }
  return vector;
}

export class VectorStore {
  constructor(dim) {
    this.dim = dim;
    this.ids = [];
    this.rows = new Map();
    this.data = new Int8Array(dim * 1024);
  }

  get size() {
    return this.ids.length;
  }

  has(id) {
    return this.rows.has(id);
  }

  set(id, vector) {
    if (vector.length !== this.dim) {
      throw new Error(`Expected a ${this.dim}-dimensional vector, got ${vector.length}`);
    }
    let row = this.rows.get(id);
    if (row === undefined) {
      row = this.ids.length;
      if ((row + 1) * this.dim > this.data.length) {
        const grown = new Int8Array(this.data.length * 2);
        grown.set(this.data);
        this.data = grown;
      }
      this.ids.push(id);
      this.rows.set(id, row);
    }
    this.data.set(vector, row * this.dim);
  }

  delete(id) {
    const row = this.rows.get(id);
    if (row === undefined) return;
    const last = this.ids.length - 1;
    if (row !== last) {
      const lastId = this.ids[last];
      this.data.copyWithin(row * this.dim, last * this.dim, (last + 1) * this.dim);
      this.ids[row] = lastId;
      this.rows.set(lastId, row);
    }
    this.ids.pop();
    this.rows.delete(id);
  }

  // Returns the k best photos, highest score first. Scores are integer dot products.
  async search(query, k, shouldStop = () => false) {
    const { data, dim } = this;
    const count = this.ids.length;
    const top = [];
    for (let row = 0; row < count; row++) {
      if (row > 0 && row % YIELD_EVERY === 0) {
        await nextTick();
        if (shouldStop()) return null;
      }
      const offset = row * dim;
      let score = 0;
      for (let i = 0; i < dim; i++) {
        score += data[offset + i] * query[i];
      }
      if (top.length < k) {
        heapPush(top, { row, score });
      } else if (score > top[0].score) {
        top[0] = { row, score };
        heapSiftDown(top);
      }
    }
    return top
      .sort((a, b) => b.score - a.score || a.row - b.row)
      .map(({ row, score }) => ({ id: this.ids[row], score }));
  }
}

function heapPush(heap, item) {
  heap.push(item);
  let child = heap.length - 1;
  while (child > 0) {
    const parent = (child - 1) >> 1;
    if (heap[parent].score <= heap[child].score) break;
    [heap[parent], heap[child]] = [heap[child], heap[parent]];
    child = parent;
  }
}

function heapSiftDown(heap) {
  let parent = 0;
  for (;;) {
    let smallest = parent;
    const left = 2 * parent + 1;
    const right = left + 1;
    if (left < heap.length && heap[left].score < heap[smallest].score) smallest = left;
    if (right < heap.length && heap[right].score < heap[smallest].score) smallest = right;
    if (smallest === parent) return;
    [heap[parent], heap[smallest]] = [heap[smallest], heap[parent]];
    parent = smallest;
  }
}
