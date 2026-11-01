import { File } from "expo-file-system";

import { decodeVector } from "./vectors";

const REQUEST_TIMEOUT_MS = 30000;
const UPLOAD_TIMEOUT_MS = 90000;

export class ApiError extends Error {
  constructor(kind, message) {
    super(message);
    this.kind = kind; // "offline" | "auth" | "rejected" | "server"
  }
}

export function normalizeServerUrl(url) {
  const trimmed = url.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  return /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
}

async function request(server, path, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(`${server.url}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "ngrok-skip-browser-warning": "1",
        "X-API-Key": server.apiKey,
        ...options.headers,
      },
    });
  } catch (error) {
    console.warn(`Request to ${path} failed:`, error);
    throw new ApiError("offline", "The server can't be reached.");
  } finally {
    clearTimeout(timer);
  }
  if (response.status === 401) {
    throw new ApiError("auth", "The server rejected the API key.");
  }
  // An ngrok error page means the tunnel exists but the notebook behind it has stopped.
  if (response.status === 404 || response.status === 502 || response.status === 503) {
    throw new ApiError("offline", "The server isn't running.");
  }
  if (!response.ok) {
    const kind = response.status < 500 ? "rejected" : "server";
    throw new ApiError(kind, `The server returned an error (${response.status}).`);
  }
  try {
    return await response.json();
  } catch {
    throw new ApiError("offline", "The server isn't running.");
  }
}

export function fetchHealth(server) {
  return request(server, "/health", { method: "GET" }, REQUEST_TIMEOUT_MS);
}

// Expo's fetch doesn't accept React Native's { uri } form parts, so the request body
// is built here from the file bytes.
export function encodeImagesForm(images, boundary) {
  const encoder = new TextEncoder();
  const chunks = [];
  images.forEach((bytes, index) => {
    chunks.push(
      encoder.encode(
        `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="files"; filename="${index}.jpg"\r\n` +
          "Content-Type: image/jpeg\r\n\r\n"
      ),
      bytes,
      encoder.encode("\r\n")
    );
  });
  chunks.push(encoder.encode(`--${boundary}--\r\n`));

  const body = new Uint8Array(chunks.reduce((size, chunk) => size + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.length;
  }
  return body;
}

// files: [{ uri }]. Returns one vector per file; null means the server couldn't read it.
export async function embedImages(server, files) {
  const images = await Promise.all(
    files.map(async (file) => new Uint8Array(await new File(file.uri).arrayBuffer()))
  );
  const boundary = `SmartGallery${Date.now()}${Math.random().toString(36).slice(2)}`;
  const body = await request(
    server,
    "/embed/images",
    {
      method: "POST",
      body: encodeImagesForm(images, boundary),
      headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    },
    UPLOAD_TIMEOUT_MS
  );
  return {
    modelVersion: body.model_version,
    vectors: body.vectors.map((vector) => (vector ? decodeVector(vector) : null)),
  };
}

export async function embedText(server, query) {
  const body = await request(
    server,
    "/embed/text",
    {
      method: "POST",
      body: JSON.stringify({ queries: [query] }),
      headers: { "Content-Type": "application/json" },
    },
    REQUEST_TIMEOUT_MS
  );
  return { modelVersion: body.model_version, vector: decodeVector(body.vectors[0]) };
}
