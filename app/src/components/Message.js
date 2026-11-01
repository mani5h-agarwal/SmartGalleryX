import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { useTheme } from "../theme";

// A centred explanation for empty, error and permission states.
export default function Message({ icon, title, body, actionLabel, onAction }) {
  const theme = useTheme();
  return (
    <View style={styles.container}>
      <Ionicons name={icon} size={40} color={theme.textMuted} />
      <Text style={[styles.title, { color: theme.text }]}>{title}</Text>
      {body ? <Text style={[styles.body, { color: theme.textMuted }]}>{body}</Text> : null}
      {actionLabel ? (
        <Pressable
          onPress={onAction}
          accessibilityRole="button"
          style={[styles.button, { backgroundColor: theme.accent }]}
        >
          <Text style={[styles.buttonText, { color: theme.onAccent }]}>{actionLabel}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { alignItems: "center", paddingHorizontal: 32, paddingVertical: 56, gap: 10 },
  title: { fontSize: 18, fontWeight: "600", textAlign: "center", marginTop: 6 },
  body: { fontSize: 14.5, lineHeight: 21, textAlign: "center" },
  button: { marginTop: 12, paddingHorizontal: 20, paddingVertical: 11, borderRadius: 12 },
  buttonText: { fontSize: 15, fontWeight: "600" },
});
