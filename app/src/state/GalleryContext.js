import * as Network from "expo-network";
import { addListener } from "expo-media-library";
import * as SecureStore from "expo-secure-store";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AppState } from "react-native";

import { ApiError, embedImages, embedText, fetchHealth, normalizeServerUrl } from "../lib/api";
import * as db from "../lib/db";
import { createIndexer } from "../lib/indexer";
import {
  getPhotoPermission,
  readLibraryPage,
  readPhotosById,
  requestPhotoPermission,
  thumbnails,
} from "../lib/photos";
import { VectorStore } from "../lib/vectors";

const API_KEY_ITEM = "server-api-key";
const SERVER_URL_META = "server-url";
const MAX_RESULTS = 60;
const RECENT_QUERIES = 6;
const RECONNECT_MS = 20000;
const LIBRARY_CHANGE_DELAY_MS = 3000;

const GalleryContext = createContext(null);

const readRecentQueries = () => db.readRecentQueries(RECENT_QUERIES);

export const useGallery = () => useContext(GalleryContext);

export function GalleryProvider({ children }) {
  const [ready, setReady] = useState(false);
  const [permission, setPermission] = useState(null);
  const [server, setServer] = useState({ url: "", apiKey: "" });
  // "unconfigured" | "checking" | "online" | "offline" | "auth"
  const [connection, setConnection] = useState("checking");
  const [progress, setProgress] = useState({ phase: "idle", done: 0, total: 0 });
  const [indexedCount, setIndexedCount] = useState(0);
  const [indexError, setIndexError] = useState(null);
  const [paused, setPaused] = useState(false);
  const [wifiOnly, setWifiOnlyState] = useState(true);
  const [waitingForWifi, setWaitingForWifi] = useState(false);
  const [libraryVersion, setLibraryVersion] = useState(0);

  const live = useRef({
    store: null,
    known: new Map(),
    modelVersion: null,
    indexer: null,
    searchToken: 0,
    syncAgain: false,
  }).current;
  const syncRef = useRef(null);
  const checkServerRef = useRef(null);

  const buildStore = useCallback(async (dim) => {
    const store = new VectorStore(dim);
    const known = new Map();
    await db.readPhotos((id, modifiedAt, vector) => {
      store.set(id, vector);
      known.set(id, modifiedAt);
    });
    live.store = store;
    live.known = known;
    live.indexer = createIndexer({
      store,
      library: { readPage: readLibraryPage },
      thumbnails,
      api: { embedImages: (files) => embedImages(live.server, files) },
      storage: {
        readKnown: async () => live.known,
        readGivenUp: db.readGivenUp,
        save: async (photos) => {
          await db.savePhotos(photos);
          photos.forEach((photo) => live.known.set(photo.id, photo.modifiedAt));
          setIndexedCount(live.known.size);
        },
        remove: async (ids) => {
          await db.deletePhotos(ids);
          ids.forEach((id) => live.known.delete(id));
          setIndexedCount(live.known.size);
        },
        recordFailures: db.recordFailures,
      },
      onProgress: setProgress,
    });
    setIndexedCount(known.size);
  }, [live]);

  const resetIndex = useCallback(async (dim) => {
    await live.indexer?.stop();
    await db.clearIndex();
    await buildStore(dim);
  }, [buildStore, live]);

  const startSync = useCallback(async () => {
    if (!live.indexer || !live.modelVersion || live.pausedFlag || !live.hasPermission) return;
    if (live.indexer.isRunning) {
      live.syncAgain = true;
      return;
    }
    if (live.wifiOnly) {
      const network = await Network.getNetworkStateAsync().catch(() => null);
      const onWifi = network?.type === Network.NetworkStateType.WIFI;
      setWaitingForWifi(!onWifi);
      if (!onWifi) return;
    } else {
      setWaitingForWifi(false);
    }
    setIndexError(null);
    try {
      const { outcome } = await live.indexer.start(live.modelVersion);
      if (outcome === "stopped") setProgress({ phase: "idle", done: 0, total: 0 });
      if (live.syncAgain) {
        live.syncAgain = false;
        syncRef.current?.();
      }
    } catch (error) {
      setProgress({ phase: "idle", done: 0, total: 0 });
      if (error.kind === "model-changed") {
        checkServerRef.current?.();
      } else if (error.kind === "offline" || error.kind === "auth") {
        setConnection(error.kind);
      } else {
        setIndexError(error.message);
      }
    }
  }, [live]);
  syncRef.current = startSync;

  const checkServer = useCallback(async () => {
    const current = live.server;
    if (!current?.url || !current?.apiKey) {
      setConnection("unconfigured");
      return false;
    }
    setConnection((state) => (state === "online" ? state : "checking"));
    let health;
    try {
      health = await fetchHealth(current);
    } catch (error) {
      if (live.server !== current) return false;
      setConnection(error instanceof ApiError && error.kind === "auth" ? "auth" : "offline");
      return false;
    }
    if (live.server !== current) return false;

    // Vectors from different models can't be compared, so a new model means a new index.
    const savedVersion = await db.getMeta("model-version");
    const dim = health.dim;
    if (savedVersion && savedVersion !== health.model_version) {
      await resetIndex(dim);
    } else if (!live.store || live.store.dim !== dim) {
      await buildStore(dim);
    }
    await db.setMeta("model-version", health.model_version);
    await db.setMeta("model-dim", dim);
    live.modelVersion = health.model_version;
    setConnection("online");
    startSync();
    return true;
  }, [buildStore, live, resetIndex, startSync]);
  checkServerRef.current = checkServer;

  useEffect(() => {
    live.pausedFlag = paused;
    live.wifiOnly = wifiOnly;
    live.hasPermission = permission?.granted ?? false;
  }, [live, paused, permission, wifiOnly]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [url, apiKey, savedWifiOnly, dim, granted] = await Promise.all([
        db.getMeta(SERVER_URL_META, ""),
        SecureStore.getItemAsync(API_KEY_ITEM).catch(() => null),
        db.getMeta("wifi-only", "1"),
        db.getMeta("model-dim"),
        getPhotoPermission().catch(() => null),
      ]);
      if (cancelled) return;
      // Search keeps working on cached queries before the server has answered.
      if (dim) {
        await buildStore(Number(dim));
        live.modelVersion = await db.getMeta("model-version");
      }
      const saved = { url, apiKey: apiKey ?? "" };
      live.server = saved;
      live.wifiOnly = savedWifiOnly === "1";
      live.hasPermission = granted?.granted ?? false;
      setServer(saved);
      setWifiOnlyState(live.wifiOnly);
      setPermission(granted);
      setReady(true);
      checkServer();
    })();
    return () => {
      cancelled = true;
    };
  }, [buildStore, checkServer, live]);

  useEffect(() => {
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active") {
        getPhotoPermission().then(setPermission).catch(() => {});
        checkServer();
        setLibraryVersion((version) => version + 1);
      } else {
        live.indexer?.stop();
      }
    });
    return () => subscription.remove();
  }, [checkServer, live]);

  useEffect(() => {
    if (connection !== "offline") return undefined;
    const timer = setInterval(checkServer, RECONNECT_MS);
    return () => clearInterval(timer);
  }, [checkServer, connection]);

  useEffect(() => {
    if (!permission?.granted) return undefined;
    let timer;
    const subscription = addListener(() => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        setLibraryVersion((version) => version + 1);
        startSync();
      }, LIBRARY_CHANGE_DELAY_MS);
    });
    return () => {
      clearTimeout(timer);
      subscription.remove();
    };
  }, [permission?.granted, startSync]);

  const requestPermission = useCallback(async () => {
    const result = await requestPhotoPermission();
    live.hasPermission = result.granted;
    setPermission(result);
    if (result.granted) {
      setLibraryVersion((version) => version + 1);
      startSync();
    }
    return result;
  }, [live, startSync]);

  const connect = useCallback(async (url, apiKey) => {
    const next = { url: normalizeServerUrl(url), apiKey: apiKey.trim() };
    await live.indexer?.stop();
    await db.setMeta(SERVER_URL_META, next.url);
    await SecureStore.setItemAsync(API_KEY_ITEM, next.apiKey);
    live.server = next;
    setServer(next);
    return checkServer();
  }, [checkServer, live]);

  const setWifiOnly = useCallback(async (value) => {
    live.wifiOnly = value;
    setWifiOnlyState(value);
    await db.setMeta("wifi-only", value ? "1" : "0");
    if (value) await live.indexer?.stop();
    startSync();
  }, [live, startSync]);

  const togglePaused = useCallback(async () => {
    const next = !live.pausedFlag;
    live.pausedFlag = next;
    setPaused(next);
    if (next) {
      await live.indexer?.stop();
      setProgress({ phase: "idle", done: 0, total: 0 });
    } else {
      startSync();
    }
  }, [live, startSync]);

  const rebuildIndex = useCallback(async () => {
    if (live.store) await resetIndex(live.store.dim);
    startSync();
  }, [live, resetIndex, startSync]);

  const search = useCallback(async (rawQuery) => {
    const query = rawQuery.trim().toLowerCase();
    const token = ++live.searchToken;
    if (!live.store || live.store.size === 0) return { query, photos: [] };

    let vector = await db.readCachedQuery(query);
    if (!vector) {
      const result = await embedText(live.server, query);
      if (result.modelVersion !== live.modelVersion) {
        await checkServer();
        throw new ApiError("server", "The model was updated. Your photos are being re-indexed.");
      }
      vector = result.vector;
      await db.cacheQuery(query, vector);
    }
    const ranked = await live.store.search(vector, MAX_RESULTS, () => token !== live.searchToken);
    if (!ranked || token !== live.searchToken) return null;
    const photos = await readPhotosById(ranked.map((result) => result.id));
    return token === live.searchToken ? { query, photos } : null;
  }, [checkServer, live]);

  const value = useMemo(() => ({
    ready,
    permission,
    requestPermission,
    server,
    connection,
    connect,
    retryConnection: checkServer,
    progress,
    indexedCount,
    indexError,
    paused,
    togglePaused,
    wifiOnly,
    setWifiOnly,
    waitingForWifi,
    rebuildIndex,
    search,
    libraryVersion,
    recentQueries: readRecentQueries,
    deleteRecentQuery: db.deleteRecentQuery,
  }), [
    checkServer, connect, connection, indexError, indexedCount, libraryVersion, paused,
    permission, progress, ready, rebuildIndex, requestPermission, search, server,
    setWifiOnly, togglePaused, waitingForWifi, wifiOnly,
  ]);

  return <GalleryContext.Provider value={value}>{children}</GalleryContext.Provider>;
}
