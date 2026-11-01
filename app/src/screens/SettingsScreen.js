import { Ionicons } from "@expo/vector-icons";
import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { useGallery } from "../state/GalleryContext";
import { useTheme } from "../theme";

const CONNECTION_TEXT = {
  unconfigured: "Not connected",
  checking: "Checking…",
  online: "Connected",
  offline: "Server offline",
  auth: "API key rejected",
};

export default function SettingsScreen({ onClose }) {
  const theme = useTheme();
  const gallery = useGallery();
  const [url, setUrl] = useState(gallery.server.url);
  const [apiKey, setApiKey] = useState(gallery.server.apiKey);
  const [saving, setSaving] = useState(false);

  const connect = async () => {
    setSaving(true);
    try {
      await gallery.connect(url, apiKey);
    } finally {
      setSaving(false);
    }
  };

  const confirmRebuild = () =>
    Alert.alert(
      "Rebuild the search index?",
      "Every photo is processed again. Your photos themselves aren't changed.",
      [
        { text: "Cancel", style: "cancel" },
        { text: "Rebuild", style: "destructive", onPress: gallery.rebuildIndex },
      ]
    );

  const input = [styles.input, { backgroundColor: theme.surface, color: theme.text }];
  const online = gallery.connection === "online";

  return (
    <Modal visible animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <SafeAreaView style={[styles.screen, { backgroundColor: theme.background }]}>
        <View style={styles.header}>
          <Text style={[styles.title, { color: theme.text }]}>Settings</Text>
          <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button" accessibilityLabel="Close settings">
            <Ionicons name="close" size={26} color={theme.text} />
          </Pressable>
        </View>
        <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === "ios" ? "padding" : undefined}>
          <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
            <Text style={[styles.section, { color: theme.textMuted }]}>SERVER</Text>
            <View style={styles.statusRow}>
              <View style={[styles.dot, { backgroundColor: online ? theme.success : theme.danger }]} />
              <Text style={[styles.status, { color: theme.text }]}>{CONNECTION_TEXT[gallery.connection]}</Text>
            </View>
            <TextInput
              value={url}
              onChangeText={setUrl}
              placeholder="https://your-domain.ngrok-free.app"
              placeholderTextColor={theme.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              accessibilityLabel="Server address"
              style={input}
            />
            <TextInput
              value={apiKey}
              onChangeText={setApiKey}
              placeholder="API key"
              placeholderTextColor={theme.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
              secureTextEntry
              accessibilityLabel="API key"
              style={input}
            />
            <Pressable
              onPress={connect}
              disabled={saving || !url.trim() || !apiKey.trim()}
              accessibilityRole="button"
              style={[
                styles.button,
                { backgroundColor: theme.accent, opacity: saving || !url.trim() || !apiKey.trim() ? 0.5 : 1 },
              ]}
            >
              {saving ? (
                <ActivityIndicator color={theme.onAccent} />
              ) : (
                <Text style={[styles.buttonText, { color: theme.onAccent }]}>Save and connect</Text>
              )}
            </Pressable>

            <Text style={[styles.section, { color: theme.textMuted }]}>INDEXING</Text>
            <View style={styles.row}>
              <View style={styles.flex}>
                <Text style={[styles.rowTitle, { color: theme.text }]}>Wi-Fi only</Text>
                <Text style={[styles.rowBody, { color: theme.textMuted }]}>
                  Each photo uploads a small thumbnail of about 30 KB.
                </Text>
              </View>
              <Switch
                value={gallery.wifiOnly}
                onValueChange={gallery.setWifiOnly}
                accessibilityLabel="Index on Wi-Fi only"
              />
            </View>
            <View style={styles.row}>
              <Text style={[styles.flex, styles.rowTitle, { color: theme.text }]}>Photos indexed</Text>
              <Text style={[styles.rowTitle, { color: theme.textMuted }]}>
                {gallery.indexedCount.toLocaleString()}
              </Text>
            </View>
            <Pressable onPress={confirmRebuild} accessibilityRole="button" style={styles.row}>
              <Text style={[styles.rowTitle, { color: theme.danger }]}>Rebuild index</Text>
            </Pressable>

            <Text style={[styles.section, { color: theme.textMuted }]}>PRIVACY</Text>
            <Text style={[styles.rowBody, { color: theme.textMuted }]}>
              Thumbnails go to your server only to be read by the model, and the server keeps none
              of them. The search index stays on this phone.
            </Text>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  screen: { flex: 1 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  title: { fontSize: 26, fontWeight: "700" },
  content: { paddingHorizontal: 20, paddingBottom: 40, gap: 10 },
  section: { fontSize: 12.5, fontWeight: "600", letterSpacing: 0.6, marginTop: 20 },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  dot: { width: 9, height: 9, borderRadius: 5 },
  status: { fontSize: 15 },
  input: { height: 48, borderRadius: 12, paddingHorizontal: 14, fontSize: 15 },
  button: { height: 48, borderRadius: 12, alignItems: "center", justifyContent: "center" },
  buttonText: { fontSize: 15.5, fontWeight: "600" },
  row: { flexDirection: "row", alignItems: "center", gap: 12, minHeight: 44 },
  rowTitle: { fontSize: 15.5 },
  rowBody: { fontSize: 13.5, lineHeight: 19, marginTop: 2 },
});
