// Brings the on-device index in line with the photo library. Dependencies are passed
// in, so the sync rules can be tested without a phone.

const BATCH_SIZE = 16;
const PARALLEL_UPLOADS = 2;
const SCAN_PAGE_SIZE = 1000;
// Decoded photos are the app's largest memory cost, so this limit spans all uploads.
const PARALLEL_THUMBNAILS = 4;

function createLimiter(limit) {
  let active = 0;
  const waiting = [];
  return async (task) => {
    if (active < limit) {
      active += 1;
    } else {
      await new Promise((resolve) => waiting.push(resolve));
    }
    try {
      return await task();
    } finally {
      const next = waiting.shift();
      if (next) next();
      else active -= 1;
    }
  };
}

export function createIndexer({ library, thumbnails, api, storage, store, onProgress }) {
  let running = null;
  let stopRequested = false;
  const limitThumbnails = createLimiter(PARALLEL_THUMBNAILS);

  async function scan() {
    const known = await storage.readKnown();
    const givenUp = await storage.readGivenUp();
    const todo = [];
    const seen = new Set();
    for (let offset = 0; ; offset += SCAN_PAGE_SIZE) {
      const page = await library.readPage(offset, SCAN_PAGE_SIZE);
      for (const photo of page) {
        seen.add(photo.id);
        if (known.get(photo.id) === photo.modifiedAt) continue;
        if (givenUp.get(photo.id) === photo.modifiedAt) continue;
        todo.push(photo);
      }
      if (page.length < SCAN_PAGE_SIZE) break;
      if (stopRequested) return null;
    }
    // An empty read is far more likely a permission change or a failed read than a wiped
    // library, and rebuilding the index would re-upload every photo.
    const removed = seen.size ? [...known.keys()].filter((id) => !seen.has(id)) : [];
    return { todo, removed, skipped: givenUp.size };
  }

  async function embedBatch(batch, serverVersion, finish) {
    const prepared = [];
    const unreadable = [];
    await Promise.all(
      batch.map((photo) =>
        limitThumbnails(async () => {
          try {
            prepared.push({ photo, uri: await thumbnails.create(photo.id) });
          } catch {
            // iCloud-only or corrupt photos land here.
            unreadable.push(photo);
          }
        })
      )
    );
    try {
      if (prepared.length) {
        const { modelVersion, vectors } = await api.embedImages(prepared);
        if (modelVersion !== serverVersion) {
          throw Object.assign(new Error("The model changed."), { kind: "model-changed" });
        }
        const done = [];
        prepared.forEach(({ photo }, index) => {
          if (vectors[index]) {
            done.push({ id: photo.id, modifiedAt: photo.modifiedAt, vector: vectors[index] });
          } else {
            unreadable.push(photo);
          }
        });
        if (done.length) {
          await storage.save(done);
          done.forEach(({ id, vector }) => store.set(id, vector));
        }
      }
      if (unreadable.length) await storage.recordFailures(unreadable);
      finish(batch.length);
    } finally {
      await thumbnails.release(prepared.map(({ uri }) => uri));
    }
  }

  async function sync(serverVersion) {
    onProgress({ phase: "scanning", done: 0, total: 0 });
    const plan = await scan();
    if (!plan) return { outcome: "stopped" };

    if (plan.removed.length) {
      await storage.remove(plan.removed);
      plan.removed.forEach((id) => store.delete(id));
    }

    const { todo } = plan;
    const total = todo.length;
    let done = 0;
    onProgress({ phase: total ? "indexing" : "idle", done, total });

    let next = 0;
    const worker = async () => {
      while (next < total && !stopRequested) {
        const batch = todo.slice(next, next + BATCH_SIZE);
        next += BATCH_SIZE;
        try {
          await embedBatch(batch, serverVersion, (count) => {
            done += count;
            onProgress({ phase: "indexing", done, total });
          });
        } catch (error) {
          stopRequested = true;
          throw error;
        }
      }
    };
    // allSettled, so no worker is still running when this sync reports it has ended.
    const results = await Promise.allSettled(Array.from({ length: PARALLEL_UPLOADS }, worker));
    const failure = results.find((result) => result.status === "rejected");
    if (failure) throw failure.reason;

    if (stopRequested) return { outcome: "stopped" };
    onProgress({ phase: "idle", done: total, total });
    return { outcome: "complete", indexed: total, removed: plan.removed.length };
  }

  return {
    get isRunning() {
      return running !== null;
    },

    // Rejects with the API error when the server fails part way; everything finished
    // so far is already saved, so the next run picks up where this one stopped.
    start(serverVersion) {
      if (running) return running;
      stopRequested = false;
      running = sync(serverVersion).finally(() => {
        running = null;
      });
      return running;
    },

    async stop() {
      stopRequested = true;
      if (running) await running.catch(() => {});
    },
  };
}
