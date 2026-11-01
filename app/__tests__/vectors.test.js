import { decodeVector, VectorStore } from "../src/lib/vectors";

const vector = (...values) => Int8Array.from(values);

describe("decodeVector", () => {
  it("reads base64 as signed bytes, matching the server's int8 encoding", () => {
    // Python: base64.b64encode(np.array([1, -1, 127, -127, 0], dtype=np.int8).tobytes())
    expect(Array.from(decodeVector("Af9/gQA="))).toEqual([1, -1, 127, -127, 0]);
  });
});

describe("VectorStore", () => {
  it("ranks by dot product, best first", async () => {
    const store = new VectorStore(2);
    store.set("sideways", vector(100, 0));
    store.set("match", vector(0, 100));
    store.set("opposite", vector(0, -100));
    const results = await store.search(vector(0, 1), 2);
    expect(results.map((r) => r.id)).toEqual(["match", "sideways"]);
  });

  it("finds the true top k over a large random set", async () => {
    const store = new VectorStore(8);
    const scores = [];
    for (let i = 0; i < 5000; i++) {
      const v = Int8Array.from({ length: 8 }, () => Math.floor(Math.random() * 255) - 127);
      store.set(`p${i}`, v);
      scores.push({ id: `p${i}`, score: v.reduce((sum, x, d) => sum + x * (d + 1), 0) });
    }
    const query = vector(1, 2, 3, 4, 5, 6, 7, 8);
    const expected = scores.sort((a, b) => b.score - a.score).slice(0, 25);
    const results = await store.search(query, 25);
    expect(results.map((r) => r.score)).toEqual(expected.map((r) => r.score));
  });

  it("grows past its initial capacity without losing vectors", async () => {
    const store = new VectorStore(2);
    for (let i = 0; i < 3000; i++) store.set(`p${i}`, vector(1, i % 100));
    store.set("best", vector(127, 127));
    expect(store.size).toBe(3001);
    expect((await store.search(vector(1, 1), 1))[0].id).toBe("best");
  });

  it("replaces a photo's vector instead of duplicating it", async () => {
    const store = new VectorStore(2);
    store.set("a", vector(1, 0));
    store.set("a", vector(0, 1));
    expect(store.size).toBe(1);
    expect(await store.search(vector(0, 1), 1)).toEqual([{ id: "a", score: 1 }]);
  });

  it("keeps the remaining vectors correct after a delete", async () => {
    const store = new VectorStore(2);
    store.set("a", vector(1, 0));
    store.set("b", vector(0, 2));
    store.set("c", vector(3, 3));
    store.delete("a");
    store.delete("missing");
    expect(store.has("a")).toBe(false);
    expect(await store.search(vector(1, 1), 5)).toEqual([
      { id: "c", score: 6 },
      { id: "b", score: 2 },
    ]);
  });

  it("rejects a vector of the wrong size", () => {
    expect(() => new VectorStore(2).set("a", vector(1, 2, 3))).toThrow();
  });

  it("returns nothing from an empty store", async () => {
    expect(await new VectorStore(2).search(vector(1, 1), 5)).toEqual([]);
  });

  it("stops early when asked, so a newer search can replace it", async () => {
    const store = new VectorStore(2);
    for (let i = 0; i < 5000; i++) store.set(`p${i}`, vector(1, 1));
    expect(await store.search(vector(1, 1), 5, () => true)).toBeNull();
  });
});
