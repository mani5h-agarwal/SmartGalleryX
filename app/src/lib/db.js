import * as SQLite from "expo-sqlite";

const PAGE_SIZE = 2000;
const MAX_ATTEMPTS = 3;
const MAX_CACHED_QUERIES = 500;

let database;

export async function openDb() {
  if (database) return database;
  const db = await SQLite.openDatabaseAsync("smart-gallery.db");
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS photos (
      id TEXT PRIMARY KEY,
      modified_at REAL NOT NULL,
      vector BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS failed (
      id TEXT PRIMARY KEY,
      modified_at REAL NOT NULL,
      attempts INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS queries (
      query TEXT PRIMARY KEY,
      vector BLOB NOT NULL,
      used_at REAL NOT NULL
    );
  `);
  database = db;
  return db;
}

export async function getMeta(key, fallback = null) {
  const db = await openDb();
  const row = await db.getFirstAsync("SELECT value FROM meta WHERE key = ?", key);
  return row ? row.value : fallback;
}

export async function setMeta(key, value) {
  const db = await openDb();
  await db.runAsync(
    "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    key,
    String(value)
  );
}

// Pages through every stored photo, passing each one to onPhoto.
export async function readPhotos(onPhoto) {
  const db = await openDb();
  let last = "";
  for (;;) {
    const rows = await db.getAllAsync(
      "SELECT id, modified_at, vector FROM photos WHERE id > ? ORDER BY id LIMIT ?",
      last,
      PAGE_SIZE
    );
    for (const row of rows) {
      const bytes = row.vector;
      onPhoto(row.id, row.modified_at, new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    }
    if (rows.length < PAGE_SIZE) return;
    last = rows[rows.length - 1].id;
  }
}

export async function savePhotos(photos) {
  const db = await openDb();
  await db.withTransactionAsync(async () => {
    for (const { id, modifiedAt, vector } of photos) {
      await db.runAsync(
        "INSERT OR REPLACE INTO photos (id, modified_at, vector) VALUES (?, ?, ?)",
        id,
        modifiedAt,
        new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength)
      );
      await db.runAsync("DELETE FROM failed WHERE id = ?", id);
    }
  });
}

export async function deletePhotos(ids) {
  const db = await openDb();
  await db.withTransactionAsync(async () => {
    for (const id of ids) {
      await db.runAsync("DELETE FROM photos WHERE id = ?", id);
      await db.runAsync("DELETE FROM failed WHERE id = ?", id);
    }
  });
}

export async function recordFailures(photos) {
  const db = await openDb();
  await db.withTransactionAsync(async () => {
    for (const { id, modifiedAt } of photos) {
      await db.runAsync(
        `INSERT INTO failed (id, modified_at, attempts) VALUES (?, ?, 1)
         ON CONFLICT(id) DO UPDATE SET
           attempts = CASE WHEN modified_at = excluded.modified_at THEN attempts + 1 ELSE 1 END,
           modified_at = excluded.modified_at`,
        id,
        modifiedAt
      );
    }
  });
}

// Photos that keep failing (a corrupt file, say) stop being retried until they change.
export async function readGivenUp() {
  const db = await openDb();
  const rows = await db.getAllAsync(
    "SELECT id, modified_at FROM failed WHERE attempts >= ?",
    MAX_ATTEMPTS
  );
  return new Map(rows.map((row) => [row.id, row.modified_at]));
}

export async function clearIndex() {
  const db = await openDb();
  await db.execAsync("DELETE FROM photos; DELETE FROM failed; DELETE FROM queries;");
}

export async function readCachedQuery(query) {
  const db = await openDb();
  const row = await db.getFirstAsync("SELECT vector FROM queries WHERE query = ?", query);
  if (!row) return null;
  await db.runAsync("UPDATE queries SET used_at = ? WHERE query = ?", Date.now(), query);
  const bytes = row.vector;
  return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice();
}

export async function cacheQuery(query, vector) {
  const db = await openDb();
  await db.runAsync(
    "INSERT OR REPLACE INTO queries (query, vector, used_at) VALUES (?, ?, ?)",
    query,
    new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength),
    Date.now()
  );
  await db.runAsync(
    `DELETE FROM queries WHERE query NOT IN
       (SELECT query FROM queries ORDER BY used_at DESC LIMIT ?)`,
    MAX_CACHED_QUERIES
  );
}

export async function readRecentQueries(limit) {
  const db = await openDb();
  const rows = await db.getAllAsync(
    "SELECT query FROM queries ORDER BY used_at DESC LIMIT ?",
    limit
  );
  return rows.map((row) => row.query);
}

export async function deleteRecentQuery(query) {
  const db = await openDb();
  await db.runAsync("DELETE FROM queries WHERE query = ?", query);
}
