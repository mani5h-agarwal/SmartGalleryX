import { FlashList } from "@shopify/flash-list";
import { Image } from "expo-image";
import { memo, useCallback } from "react";
import { Pressable, StyleSheet, useWindowDimensions, View } from "react-native";

import { useTheme } from "../theme";

const COLUMNS = 3;
const GAP = 2;

const PhotoTile = memo(function PhotoTile({ photo, index, size, onPress, placeholder }) {
  return (
    <Pressable
      onPress={() => onPress(index)}
      accessibilityRole="imagebutton"
      accessibilityLabel={`Photo ${index + 1}`}
      style={{ width: size, height: size, padding: GAP / 2 }}
    >
      <Image
        source={{ uri: photo.uri }}
        recyclingKey={photo.id}
        contentFit="cover"
        cachePolicy="memory-disk"
        transition={100}
        style={[styles.image, { backgroundColor: placeholder }]}
      />
    </Pressable>
  );
});

export default function PhotoGrid({ photos, onPressPhoto, onEndReached, ListHeaderComponent, ListFooterComponent, ListEmptyComponent }) {
  const theme = useTheme();
  const { width } = useWindowDimensions();
  const size = width / COLUMNS;

  const renderItem = useCallback(
    ({ item, index }) => (
      <PhotoTile
        photo={item}
        index={index}
        size={size}
        onPress={onPressPhoto}
        placeholder={theme.surface}
      />
    ),
    [onPressPhoto, size, theme.surface]
  );

  return (
    <View style={styles.list}>
      <FlashList
        data={photos}
        renderItem={renderItem}
        keyExtractor={(photo) => photo.id}
        numColumns={COLUMNS}
        keyboardDismissMode="on-drag"
        keyboardShouldPersistTaps="handled"
        onEndReached={onEndReached}
        onEndReachedThreshold={1}
        ListHeaderComponent={ListHeaderComponent}
        ListFooterComponent={ListFooterComponent}
        ListEmptyComponent={ListEmptyComponent}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  list: { flex: 1 },
  image: { flex: 1 },
});
