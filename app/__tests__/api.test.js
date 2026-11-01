import { ApiError, embedImages, embedText, encodeImagesForm, fetchHealth, normalizeServerUrl } from "../src/lib/api";

jest.mock("expo-file-system", () => ({
  File: class {
    constructor(uri) {
      this.uri = uri;
    }
    async arrayBuffer() {
      return Uint8Array.from([0xff, 0xd8, this.uri.length]).buffer;
    }
  },
}));

const server = { url: "https://example.test", apiKey: "secret" };
const respond = (status, body) => ({ ok: status < 400, status, json: async () => body });

afterEach(() => {
  delete global.fetch;
});

describe("normalizeServerUrl", () => {
  it.each([
    ["  abc.ngrok-free.app/ ", "https://abc.ngrok-free.app"],
    ["http://10.0.0.2:8000//", "http://10.0.0.2:8000"],
    ["   ", ""],
  ])("cleans %j", (input, expected) => {
    expect(normalizeServerUrl(input)).toBe(expected);
  });
});

describe("encodeImagesForm", () => {
  it("keeps binary data intact and separates each file", () => {
    const body = encodeImagesForm([Uint8Array.from([0, 255, 13, 10]), Uint8Array.from([1])], "B");
    expect(new TextDecoder("latin1").decode(body)).toBe(
      '--B\r\nContent-Disposition: form-data; name="files"; filename="0.jpg"\r\n' +
        "Content-Type: image/jpeg\r\n\r\n\u0000\u00ff\r\n\r\n" +
        '--B\r\nContent-Disposition: form-data; name="files"; filename="1.jpg"\r\n' +
        "Content-Type: image/jpeg\r\n\r\n\u0001\r\n--B--\r\n"
    );
  });
});

describe("requests", () => {
  it("sends the API key and the ngrok bypass header", async () => {
    global.fetch = jest.fn(async () => respond(200, { status: "ok" }));
    await fetchHealth(server);
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://example.test/health");
    expect(options.headers).toMatchObject({ "X-API-Key": "secret", "ngrok-skip-browser-warning": "1" });
  });

  it("decodes image vectors and keeps nulls for unreadable images", async () => {
    global.fetch = jest.fn(async () => respond(200, { model_version: "v1", vectors: ["Af8=", null] }));
    const { modelVersion, vectors } = await embedImages(server, [{ uri: "file://a.jpg" }, { uri: "file://b.jpg" }]);
    expect(modelVersion).toBe("v1");
    expect(Array.from(vectors[0])).toEqual([1, -1]);
    expect(vectors[1]).toBeNull();
    expect(global.fetch.mock.calls[0][0]).toBe("https://example.test/embed/images");
  });

  // Expo's fetch throws on React Native's { uri } form parts, so these must stay raw bytes.
  it("uploads images as a raw multipart body that expo/fetch accepts", async () => {
    global.fetch = jest.fn(async () => respond(200, { model_version: "v1", vectors: ["Af8="] }));
    await embedImages(server, [{ uri: "file://a.jpg" }]);
    const [, options] = global.fetch.mock.calls[0];
    expect(options.body).toBeInstanceOf(Uint8Array);
    const boundary = options.headers["Content-Type"].match(/^multipart\/form-data; boundary=(.+)$/)[1];
    expect(new TextDecoder("latin1").decode(options.body)).toContain(`--${boundary}--\r\n`);
  });

  it("posts the query in the server's format", async () => {
    global.fetch = jest.fn(async () => respond(200, { model_version: "v1", vectors: ["Af8="] }));
    const { vector } = await embedText(server, "a dog");
    const [, options] = global.fetch.mock.calls[0];
    expect(JSON.parse(options.body)).toEqual({ queries: ["a dog"] });
    expect(Array.from(vector)).toEqual([1, -1]);
  });

  it.each([
    [401, "auth"],
    [404, "offline"],
    [502, "offline"],
    [413, "rejected"],
    [500, "server"],
  ])("maps HTTP %i to the %s error", async (status, kind) => {
    global.fetch = jest.fn(async () => respond(status, {}));
    await expect(fetchHealth(server)).rejects.toMatchObject({ kind });
  });

  it("treats a network failure as offline", async () => {
    global.fetch = jest.fn(async () => {
      throw new TypeError("Network request failed");
    });
    await expect(fetchHealth(server)).rejects.toBeInstanceOf(ApiError);
    await expect(fetchHealth(server)).rejects.toMatchObject({ kind: "offline" });
  });

  it("treats an HTML tunnel page as offline", async () => {
    global.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError("Unexpected token <");
      },
    }));
    await expect(fetchHealth(server)).rejects.toMatchObject({ kind: "offline" });
  });
});
