import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { useGallery } from "../state/GalleryContext";
import { useTheme } from "../theme";

function describe({ connection, progress, paused, waitingForWifi, indexError, indexedCount }) {
  if (connection === "unconfigured") {
    return { icon: "server-outline", text: "Connect your server to search photos", action: "settings" };
  }
  if (connection === "auth") {
    return { icon: "key-outline", text: "The server rejected your API key", action: "settings", danger: true };
  }
  if (connection === "offline") {
    return {
      icon: "cloud-offline-outline",
      text: indexedCount
        ? "Server offline. Recent searches still work."
        : "Server offline. Start the Kaggle notebook.",
      action: "settings",
    };
  }
  if (indexError) return { icon: "alert-circle-outline", text: indexError, danger: true };
  if (paused) return { icon: "pause-circle-outline", text: "Indexing paused", action: "toggle", actionIcon: "play" };
  if (waitingForWifi) return { icon: "wifi-outline", text: "Indexing resumes on Wi-Fi" };
  if (progress.phase === "indexing" && progress.total > 0) {
    return {
      icon: "sparkles-outline",
      text: `Indexing ${progress.done.toLocaleString()} of ${progress.total.toLocaleString()} photos`,
      action: "toggle",
      actionIcon: "pause",
      progress: progress.done / progress.total,
    };
  }
  return null;
}

export default function StatusBanner({ onOpenSettings }) {
  const theme = useTheme();
  const gallery = useGallery();
  const status = describe(gallery);
  if (!status) return null;

  const tappable = status.action === "settings";
  const color = status.danger ? theme.danger : theme.textMuted;
  return (
    <Pressable
      disabled={!tappable}
      onPress={onOpenSettings}
      accessibilityRole={tappable ? "button" : "text"}
      style={[styles.banner, { backgroundColor: status.danger ? theme.dangerSoft : theme.surface }]}
    >
      <View style={styles.row}>
        <Ionicons name={status.icon} size={18} color={color} />
        <Text numberOfLines={2} style={[styles.text, { color: status.danger ? theme.danger : theme.text }]}>
          {status.text}
        </Text>
        {status.action === "toggle" && (
          <Pressable
            onPress={gallery.togglePaused}
            hitSlop={12}
            accessibilityRole="button"
            accessibilityLabel={gallery.paused ? "Resume indexing" : "Pause indexing"}
          >
            <Ionicons name={status.actionIcon} size={18} color={theme.accent} />
          </Pressable>
        )}
        {tappable && <Ionicons name="chevron-forward" size={18} color={color} />}
      </View>
      {status.progress !== undefined && (
        <View style={[styles.track, { backgroundColor: theme.surfaceStrong }]}>
          <View
            style={[
              styles.fill,
              { backgroundColor: theme.accent, width: `${Math.round(status.progress * 100)}%` },
            ]}
          />
        </View>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  banner: { borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10, gap: 8 },
  row: { flexDirection: "row", alignItems: "center", gap: 10 },
  text: { flex: 1, fontSize: 13.5 },
  track: { height: 3, borderRadius: 2, overflow: "hidden" },
  fill: { height: "100%" },
});
