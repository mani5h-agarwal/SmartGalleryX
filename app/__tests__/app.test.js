import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";

import { SafeAreaProvider } from "react-native-safe-area-context";

import { GalleryProvider } from "../src/state/GalleryContext";
import GalleryScreen from "../src/screens/GalleryScreen";

const mockPhotos = [
  { id: "1", uri: "file://1.jpg", filename: "1.jpg", width: 10, height: 10, creationTime: 1 },
  { id: "2", uri: "file://2.jpg", filename: "2.jpg", width: 10, height: 10, creationTime: 2 },
];
const mockRows = new Map();
const mockMeta = new Map();
let mockCachedQuery = null;

jest.mock("@expo/vector-icons", () => ({ Ionicons: () => null }));
jest.mock("expo-image", () => ({ Image: () => null }));
jest.mock("expo-haptics", () => ({ selectionAsync: jest.fn(async () => {}) }));
jest.mock("expo-sharing", () => ({ shareAsync: jest.fn() }));
jest.mock("expo-network", () => ({
  NetworkStateType: { WIFI: "WIFI" },
  getNetworkStateAsync: jest.fn(async () => ({ type: "WIFI" })),
}));
jest.mock("expo-media-library", () => ({ addListener: () => ({ remove() {} }) }));
jest.mock("expo-secure-store", () => ({
  getItemAsync: jest.fn(async () => "key"),
  setItemAsync: jest.fn(async () => {}),
}));
jest.mock("@shopify/flash-list", () => {
  const { View } = require("react-native");
  return {
    FlashList: ({ data, renderItem, ListHeaderComponent, ListEmptyComponent }) => (
      <View>
        {ListHeaderComponent}
        {data.length ? data.map((item, index) => <View key={item.id}>{renderItem({ item, index })}</View>) : ListEmptyComponent}
      </View>
    ),
  };
});
jest.mock("../src/lib/photos", () => ({
  getPhotoPermission: jest.fn(async () => mockPermission()),
  requestPhotoPermission: jest.fn(async () => ({ granted: true })),
  readLibraryPage: jest.fn(async () => [...mockRows.keys()].map((id) => ({ id, modifiedAt: 1 }))),
  readGalleryPage: jest.fn(async () => mockPhotos),
  readPhotosById: jest.fn(async (ids) => mockPhotos.filter((photo) => ids.includes(photo.id))),
  thumbnails: { create: jest.fn(), release: jest.fn() },
  exportForSharing: jest.fn(),
}));
jest.mock("../src/lib/db", () => ({
  getMeta: jest.fn(async (key, fallback = null) => mockMeta.get(key) ?? fallback),
  setMeta: jest.fn(async (key, value) => void mockMeta.set(key, String(value))),
  readPhotos: jest.fn(async (onPhoto) => mockRows.forEach((vector, id) => onPhoto(id, 1, vector))),
  savePhotos: jest.fn(),
  deletePhotos: jest.fn(),
  recordFailures: jest.fn(),
  readGivenUp: jest.fn(async () => new Map()),
  clearIndex: jest.fn(async () => mockRows.clear()),
  readCachedQuery: jest.fn(async () => mockCachedQuery),
  cacheQuery: jest.fn(async () => {}),
  readRecentQueries: jest.fn(async () => ["my old search"]),
  deleteRecentQuery: jest.fn(async () => {}),
}));

let granted;
const mockPermission = () => ({ granted, canAskAgain: true });

const health = { status: "ok", model_version: "v1", dim: 2, model: "m", quant_scale: 512, max_files: 32 };
const respond = (body) => ({ ok: true, status: 200, json: async () => body });

const metrics = { frame: { x: 0, y: 0, width: 390, height: 844 }, insets: { top: 0, left: 0, right: 0, bottom: 0 } };

const renderApp = () =>
  render(
    <SafeAreaProvider initialMetrics={metrics}>
      <GalleryProvider>
        <GalleryScreen />
      </GalleryProvider>
    </SafeAreaProvider>
  );

beforeEach(() => {
  granted = true;
  mockCachedQuery = null;
  mockRows.clear();
  mockMeta.clear();
  mockMeta.set("server-url", "https://example.test");
  global.fetch = jest.fn(async (url) => {
    if (url.endsWith("/health")) return respond(health);
    if (url.endsWith("/embed/text")) return respond({ model_version: "v1", vectors: ["Af8="] });
    throw new Error(`Unexpected request to ${url}`);
  });
});

afterEach(() => {
  delete global.fetch;
});

describe("Smart Gallery", () => {
  it("asks for photo access first", async () => {
    granted = false;
    await renderApp();
    expect(await screen.findByText("Allow access")).toBeTruthy();
  });

  it("shows the library once it has access", async () => {
    await renderApp();
    expect(await screen.findByLabelText("Photo 2")).toBeTruthy();
  });

  it("prompts for setup when no server is saved", async () => {
    mockMeta.delete("server-url");
    await renderApp();
    expect(await screen.findByText("Connect your server to search photos")).toBeTruthy();
  });

  it("says when the server is offline", async () => {
    global.fetch = jest.fn(async () => {
      throw new TypeError("Network request failed");
    });
    await renderApp();
    expect(await screen.findByText("Server offline. Start the Kaggle notebook.")).toBeTruthy();
  });

  it("ranks indexed photos on the device and shows them", async () => {
    mockMeta.set("model-dim", "2");
    mockMeta.set("model-version", "v1");
    mockRows.set("1", Int8Array.from([-5, 5]));
    mockRows.set("2", Int8Array.from([5, -5]));
    await renderApp();
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const input = await screen.findByLabelText("Search photos");
    await fireEvent.changeText(input, "A Dog");
    await fireEvent(await screen.findByLabelText("Search photos"), "submitEditing");
    expect(await screen.findByText(/Best 2 matches from 2 indexed photos/)).toBeTruthy();
    const body = JSON.parse(global.fetch.mock.calls.find(([url]) => url.endsWith("/embed/text"))[1].body);
    expect(body.queries).toEqual(["a dog"]);
  });

  it("serves a repeat search offline from its cached vector", async () => {
    mockMeta.set("model-dim", "2");
    mockMeta.set("model-version", "v1");
    mockRows.set("1", Int8Array.from([5, -5]));
    mockCachedQuery = Int8Array.from([1, -1]);
    global.fetch = jest.fn(async () => {
      throw new TypeError("Network request failed");
    });
    await renderApp();
    const input = await screen.findByLabelText("Search photos");
    await fireEvent.changeText(input, "a dog");
    await fireEvent(await screen.findByLabelText("Search photos"), "submitEditing");
    expect(await screen.findByText(/Best 1 match from 1 indexed photos/)).toBeTruthy();
  });

  it("shows a retry when search fails", async () => {
    mockMeta.set("model-dim", "2");
    mockMeta.set("model-version", "v1");
    mockRows.set("1", Int8Array.from([5, -5]));
    await renderApp();
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    global.fetch = jest.fn(async () => {
      throw new TypeError("Network request failed");
    });
    const input = await screen.findByLabelText("Search photos");
    await fireEvent.changeText(input, "a dog");
    await fireEvent(await screen.findByLabelText("Search photos"), "submitEditing");
    expect(await screen.findByText("Search isn't available")).toBeTruthy();
    expect(screen.getByText("Try again")).toBeTruthy();
  });

  it("clears the index when the server has a new model", async () => {
    mockMeta.set("model-dim", "2");
    mockMeta.set("model-version", "old");
    mockRows.set("1", Int8Array.from([5, -5]));
    await renderApp();
    await waitFor(() => expect(mockMeta.get("model-version")).toBe("v1"));
    expect(mockRows.size).toBe(0);
  });

  it("opens settings with the saved server filled in", async () => {
    await renderApp();
    await fireEvent.press(await screen.findByLabelText("Settings"));
    expect(await screen.findByDisplayValue("https://example.test")).toBeTruthy();
  });

  it("opens a photo in the viewer", async () => {
    await renderApp();
    await fireEvent.press(await screen.findByLabelText("Photo 1"));
    expect(await screen.findByLabelText("Share")).toBeTruthy();
  });
});
