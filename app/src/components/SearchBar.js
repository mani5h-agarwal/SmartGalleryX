import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, TextInput, View } from "react-native";

import { useTheme } from "../theme";

export default function SearchBar({ value, onChangeText, onSubmit, onClear, onFocus, onBlur, inputRef }) {
  const theme = useTheme();
  return (
    <View style={[styles.bar, { backgroundColor: theme.surface }]}>
      <Ionicons name="search" size={20} color={theme.textMuted} />
      <TextInput
        ref={inputRef}
        value={value}
        onChangeText={onChangeText}
        onSubmitEditing={onSubmit}
        onFocus={onFocus}
        onBlur={onBlur}
        placeholder="Describe a photo…"
        placeholderTextColor={theme.textMuted}
        selectionColor={theme.accent}
        returnKeyType="search"
        autoCorrect
        maxLength={300}
        accessibilityLabel="Search photos"
        style={[styles.input, { color: theme.text }]}
      />
      {value.length > 0 && (
        <Pressable onPress={onClear} hitSlop={12} accessibilityRole="button" accessibilityLabel="Clear search">
          <Ionicons name="close-circle" size={20} color={theme.textMuted} />
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    height: 48,
    borderRadius: 14,
    paddingHorizontal: 14,
  },
  input: { flex: 1, fontSize: 16, height: "100%" },
});
