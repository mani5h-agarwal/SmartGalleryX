import { createIndexer } from "../src/lib/indexer";
import { VectorStore } from "../src/lib/vectors";

const VERSION = "v1";

function setup({ photos, known = new Map(), givenUp = new Map(), embedImages } = {}) {
  const saved = new Map(known);
  const calls = { failures: [], released: [], uploaded: 0, removed: [], thumbnailsNow: 0, thumbnailPeak: 0 };
  const store = new VectorStore(2);
  known.forEach((_, id) => store.set(id, Int8Array.from([1, 1])));
  const progress = [];
  const indexer = createIndexer({
    store,
    library: {
      readPage: async (offset, limit) => photos.slice(offset, offset + limit),
    },
    thumbnails: {
      create: async (id) => {
        calls.thumbnailsNow += 1;
        calls.thumbnailPeak = Math.max(calls.thumbnailPeak, calls.thumbnailsNow);
        await new Promise((resolve) => setTimeout(resolve, 1));
        calls.thumbnailsNow -= 1;
        if (id.startsWith("cloud")) throw new Error("not downloaded");
        return `file://${id}.jpg`;
      },
      release: async (uris) => calls.released.push(...uris),
    },
    api: {
      embedImages:
        embedImages ??
        (async (files) => {
          calls.uploaded += files.length;
          return { modelVersion: VERSION, vectors: files.map(() => Int8Array.from([1, 2])) };
        }),
    },
    storage: {
      readKnown: async () => new Map(saved),
      readGivenUp: async () => givenUp,
      save: async (items) => items.forEach((p) => saved.set(p.id, p.modifiedAt)),
      remove: async (ids) => {
        ids.forEach((id) => saved.delete(id));
        calls.removed.push(...ids);
      },
      recordFailures: async (items) => calls.failures.push(...items.map((p) => p.id)),
    },
    onProgress: (update) => progress.push(update),
  });
  return { indexer, saved, calls, store, progress };
}

const photo = (id, modifiedAt = 1) => ({ id, modifiedAt });
const many = (count) => Array.from({ length: count }, (_, i) => photo(`p${i}`));

describe("indexer", () => {
  it("indexes every new photo across batches and reports progress", async () => {
    const { indexer, saved, store, calls, progress } = setup({ photos: many(50) });
    const result = await indexer.start(VERSION);
    expect(result).toMatchObject({ outcome: "complete", indexed: 50 });
    expect(saved.size).toBe(50);
    expect(store.size).toBe(50);
    expect(calls.released).toHaveLength(50);
    expect(progress.at(-1)).toEqual({ phase: "idle", done: 50, total: 50 });
  });

  // Each one decodes a full photo. Thirty-two at once got the app killed for using 3.5 GB.
  it("never builds more than four thumbnails at once, across all uploads", async () => {
    const { indexer, calls, saved } = setup({ photos: many(100) });
    await indexer.start(VERSION);
    expect(saved.size).toBe(100);
    expect(calls.thumbnailPeak).toBe(4);
  });

  it("scans libraries larger than one page", async () => {
    const { indexer, saved } = setup({ photos: many(2500) });
    await indexer.start(VERSION);
    expect(saved.size).toBe(2500);
  });

  it("uploads nothing for photos that are already indexed", async () => {
    const { indexer, calls } = setup({ photos: [photo("a")], known: new Map([["a", 1]]) });
    await indexer.start(VERSION);
    expect(calls.uploaded).toBe(0);
  });

  it("re-indexes a photo that was edited", async () => {
    const { indexer, calls, saved } = setup({ photos: [photo("a", 2)], known: new Map([["a", 1]]) });
    await indexer.start(VERSION);
    expect(calls.uploaded).toBe(1);
    expect(saved.get("a")).toBe(2);
  });

  it("drops photos deleted from the library", async () => {
    const { indexer, store, calls } = setup({
      photos: [photo("a")],
      known: new Map([["a", 1], ["gone", 1]]),
    });
    await indexer.start(VERSION);
    expect(calls.removed).toEqual(["gone"]);
    expect(store.has("gone")).toBe(false);
  });

  it("keeps the index when the library reads back empty", async () => {
    const { indexer, store, calls } = setup({ photos: [], known: new Map([["a", 1]]) });
    await indexer.start(VERSION);
    expect(calls.removed).toEqual([]);
    expect(store.has("a")).toBe(true);
  });

  it("records unreadable photos and still finishes the rest", async () => {
    const { indexer, calls, saved } = setup({ photos: [photo("cloud-1"), photo("a")] });
    await indexer.start(VERSION);
    expect(calls.failures).toEqual(["cloud-1"]);
    expect(saved.has("a")).toBe(true);
  });

  it("records a photo the server could not read", async () => {
    const { indexer, calls, saved } = setup({
      photos: [photo("a"), photo("b")],
      embedImages: async () => ({ modelVersion: VERSION, vectors: [Int8Array.from([1, 2]), null] }),
    });
    await indexer.start(VERSION);
    expect(saved.has("a")).toBe(true);
    expect(calls.failures).toHaveLength(1);
  });

  it("skips photos it has given up on, until they change", async () => {
    const { indexer, calls } = setup({
      photos: [photo("stuck", 1), photo("edited", 2)],
      givenUp: new Map([["stuck", 1], ["edited", 1]]),
    });
    await indexer.start(VERSION);
    expect(calls.uploaded).toBe(1);
  });

  it("keeps finished work and cleans up when the server fails part way", async () => {
    let batches = 0;
    const offline = Object.assign(new Error("offline"), { kind: "offline" });
    const { indexer, saved, calls } = setup({
      photos: many(64),
      embedImages: async (files) => {
        if (++batches > 2) throw offline;
        return { modelVersion: VERSION, vectors: files.map(() => Int8Array.from([1, 2])) };
      },
    });
    await expect(indexer.start(VERSION)).rejects.toBe(offline);
    expect(saved.size).toBe(32);
    expect(calls.released.length).toBeGreaterThanOrEqual(48);
    expect(indexer.isRunning).toBe(false);
  });

  it("has no upload still running once a failed sync reports it has ended", async () => {
    let active = 0;
    let activeAfterFailure = 0;
    const { indexer } = setup({
      photos: many(64),
      embedImages: async (files) => {
        active += 1;
        await new Promise((resolve) => setTimeout(resolve, active === 1 ? 1 : 20));
        active -= 1;
        if (files[0].photo.id === "p0") throw new Error("offline");
        return { modelVersion: VERSION, vectors: files.map(() => Int8Array.from([1, 2])) };
      },
    });
    await indexer.start(VERSION).catch(() => {
      activeAfterFailure = active;
    });
    expect(activeAfterFailure).toBe(0);
  });

  it("refuses vectors from a different model, so the index can't mix two models", async () => {
    const { indexer, saved } = setup({
      photos: [photo("a")],
      embedImages: async () => ({ modelVersion: "v2", vectors: [Int8Array.from([1, 2])] }),
    });
    await expect(indexer.start(VERSION)).rejects.toMatchObject({ kind: "model-changed" });
    expect(saved.size).toBe(0);
  });

  it("runs only one sync at a time", async () => {
    const { indexer, calls } = setup({ photos: many(20) });
    await Promise.all([indexer.start(VERSION), indexer.start(VERSION)]);
    expect(calls.uploaded).toBe(20);
  });

  it("stops between batches when asked and resumes on the next run", async () => {
    const { indexer, saved } = setup({ photos: many(200) });
    const run = indexer.start(VERSION);
    await indexer.stop();
    expect((await run).outcome).toBe("stopped");
    expect(saved.size).toBeLessThan(200);
    await indexer.start(VERSION);
    expect(saved.size).toBe(200);
  });
});
