import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Keyboard,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import Message from "../components/Message";
import PhotoGrid from "../components/PhotoGrid";
import SearchBar from "../components/SearchBar";
import StatusBanner from "../components/StatusBanner";
import { readGalleryPage } from "../lib/photos";
import { useGallery } from "../state/GalleryContext";
import { useTheme } from "../theme";
import PhotoViewer from "./PhotoViewer";
import SettingsScreen from "./SettingsScreen";

const PAGE_SIZE = 120;
const SUGGESTIONS = [
  "sunset at the beach",
  "my dog playing outside",
  "friends at a dinner table",
  "mountains and snow",
  "a birthday cake",
];

function Chip({ label, onPress, onRemove }) {
  const theme = useTheme();
  return (
    <View style={[styles.chip, { backgroundColor: theme.surface }]}>
      <Pressable onPress={() => onPress(label)} accessibilityRole="button">
        <Text style={[styles.chipText, { color: theme.text }]}>{label}</Text>
      </Pressable>
      {onRemove && (
        <Pressable
          onPress={() => onRemove(label)}
          hitSlop={10}
          accessibilityRole="button"
          accessibilityLabel={`Remove ${label}`}
        >
          <Ionicons name="close" size={15} color={theme.textMuted} />
        </Pressable>
      )}
    </View>
  );
}

function useLibrary(enabled, version) {
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(true);
  const paging = useRef({ offset: 0, done: false, busy: false, generation: 0 }).current;

  const loadMore = useCallback(async () => {
    if (paging.busy || paging.done) return;
    paging.busy = true;
    const { generation, offset } = paging;
    try {
      const page = await readGalleryPage(offset, PAGE_SIZE);
      if (generation !== paging.generation) return;
      paging.offset += page.length;
      paging.done = page.length < PAGE_SIZE;
      setPhotos((current) => (offset === 0 ? page : [...current, ...page]));
    } catch {
      paging.done = true;
    } finally {
      paging.busy = false;
      setLoading(false);
    }
  }, [paging]);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    paging.generation += 1;
    paging.offset = 0;
    paging.done = false;
    paging.busy = false;
    loadMore();
  }, [enabled, version, loadMore, paging]);

  return { photos, loading, loadMore };
}

export default function GalleryScreen() {
  const theme = useTheme();
  const gallery = useGallery();
  const { permission, requestPermission, connection, recentQueries, search, deleteRecentQuery } = gallery;
  const canRead = permission?.granted ?? false;
  const { photos, loading, loadMore } = useLibrary(canRead, gallery.libraryVersion);

  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [recent, setRecent] = useState([]);
  const [result, setResult] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [viewerIndex, setViewerIndex] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    if (focused) recentQueries().then(setRecent).catch(() => {});
  }, [focused, recentQueries]);

  const runSearch = useCallback(async (text) => {
    if (!text.trim()) return;
    Keyboard.dismiss();
    setQuery(text);
    setSearchError(null);
    setSearching(true);
    try {
      const found = await search(text);
      if (found) {
        setResult(found);
        Haptics.selectionAsync().catch(() => {});
      }
    } catch (error) {
      setResult(null);
      setSearchError(error.message);
    } finally {
      setSearching(false);
    }
  }, [search]);

  const clearSearch = useCallback(() => {
    setQuery("");
    setResult(null);
    setSearchError(null);
  }, []);

  const removeRecent = useCallback(async (text) => {
    await deleteRecentQuery(text);
    setRecent((items) => items.filter((item) => item !== text));
  }, [deleteRecentQuery]);

  const askPermission = useCallback(async () => {
    const response = await requestPermission();
    if (!response.granted && !response.canAskAgain) Linking.openSettings();
  }, [requestPermission]);

  const shown = result ? result.photos : photos;
  const openPhoto = useCallback((index) => setViewerIndex(index), []);
  const showPrompts = focused && !query.trim();

  if (!gallery.ready) {
    return (
      <View style={[styles.center, { backgroundColor: theme.background }]}>
        <ActivityIndicator color={theme.accent} />
      </View>
    );
  }

  if (!canRead) {
    return (
      <SafeAreaView style={[styles.screen, styles.center, { backgroundColor: theme.background }]}>
        <Message
          icon="images-outline"
          title="See and search your photos"
          body="SmartGalleryX needs access to your photos to show them and find them from a description."
          actionLabel={permission?.canAskAgain === false ? "Open Settings" : "Allow access"}
          onAction={askPermission}
        />
      </SafeAreaView>
    );
  }

  const header = (
    <>
      {result && (
        <Text style={[styles.resultSummary, { color: theme.textMuted }]}>
          {result.photos.length
            ? `Best ${result.photos.length} ${result.photos.length === 1 ? "match" : "matches"} from ${gallery.indexedCount.toLocaleString()} indexed photos`
            : ""}
        </Text>
      )}
    </>
  );

  const empty = result ? (
    <Message
      icon="search-outline"
      title="Nothing to show yet"
      body={gallery.indexedCount ? "Try describing it differently." : "Photos appear here once they've been indexed."}
    />
  ) : loading ? null : (
    <Message icon="images-outline" title="No photos yet" body="Photos you take will show up here." />
  );

  return (
    <SafeAreaView edges={["top"]} style={[styles.screen, { backgroundColor: theme.background }]}>
      <View style={styles.top}>
        <View style={styles.titleRow}>
          <Text accessibilityRole="header" style={[styles.title, { color: theme.text }]}>SmartGalleryX</Text>
          <Pressable
            onPress={() => setSettingsOpen(true)}
            hitSlop={12}
            accessibilityRole="button"
            accessibilityLabel="Settings"
          >
            <Ionicons name="settings-outline" size={24} color={theme.text} />
          </Pressable>
        </View>
        <SearchBar
          value={query}
          onChangeText={setQuery}
          onSubmit={() => runSearch(query)}
          onClear={clearSearch}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
        />
        <StatusBanner onOpenSettings={() => setSettingsOpen(true)} />
        {permission.accessPrivileges === "limited" && (
          <Text style={[styles.hint, { color: theme.textMuted }]}>
            You've shared only some photos with SmartGalleryX.
          </Text>
        )}
      </View>

      {showPrompts ? (
        <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.prompts}>
          {recent.length > 0 && (
            <>
              <Text style={[styles.promptTitle, { color: theme.textMuted }]}>Recent</Text>
              <View style={styles.chips}>
                {recent.map((item) => (
                  <Chip key={item} label={item} onPress={runSearch} onRemove={removeRecent} />
                ))}
              </View>
            </>
          )}
          <Text style={[styles.promptTitle, { color: theme.textMuted }]}>Try</Text>
          <View style={styles.chips}>
            {SUGGESTIONS.map((item) => (
              <Chip key={item} label={item} onPress={runSearch} />
            ))}
          </View>
        </ScrollView>
      ) : searching ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={theme.accent} />
        </View>
      ) : searchError ? (
        <Message
          icon={connection === "online" ? "alert-circle-outline" : "cloud-offline-outline"}
          title="Search isn't available"
          body={searchError}
          actionLabel="Try again"
          onAction={() => runSearch(query)}
        />
      ) : loading && !result ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={theme.accent} />
        </View>
      ) : (
        <PhotoGrid
          photos={shown}
          onPressPhoto={openPhoto}
          onEndReached={result ? undefined : loadMore}
          ListHeaderComponent={header}
          ListEmptyComponent={empty}
        />
      )}

      {viewerIndex !== null && (
        <PhotoViewer photos={shown} startIndex={viewerIndex} onClose={() => setViewerIndex(null)} />
      )}
      {settingsOpen && <SettingsScreen onClose={() => setSettingsOpen(false)} />}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  top: { paddingHorizontal: 14, paddingBottom: 10, gap: 10 },
  titleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 2,
    paddingTop: 6,
  },
  title: { fontSize: 28, fontWeight: "700" },
  hint: { fontSize: 12.5, paddingHorizontal: 2 },
  resultSummary: { fontSize: 13, paddingHorizontal: 16, paddingBottom: 8 },
  prompts: { paddingHorizontal: 16, paddingBottom: 24, gap: 10 },
  promptTitle: { fontSize: 13.5, fontWeight: "600", marginTop: 8 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 18,
    paddingHorizontal: 13,
    paddingVertical: 9,
  },
  chipText: { fontSize: 14 },
});
