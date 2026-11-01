import { Ionicons } from "@expo/vector-icons";
import { Image } from "expo-image";
import * as Sharing from "expo-sharing";
import { useCallback, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { exportForSharing } from "../lib/photos";

const formatDate = (timestamp) =>
  timestamp
    ? new Date(timestamp).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : "Unknown date";

function ZoomablePhoto({ photo, width, height }) {
  return (
    <ScrollView
      style={{ width, height }}
      contentContainerStyle={{ width, height }}
      minimumZoomScale={1}
      maximumZoomScale={4}
      showsVerticalScrollIndicator={false}
      showsHorizontalScrollIndicator={false}
      centerContent
    >
      <Image
        source={{ uri: photo.uri }}
        contentFit="contain"
        transition={120}
        style={{ width, height }}
      />
    </ScrollView>
  );
}

export default function PhotoViewer({ photos, startIndex, onClose }) {
  const { width, height } = useWindowDimensions();
  const insets = useSafeAreaInsets();
  const [index, setIndex] = useState(startIndex);
  const [sharing, setSharing] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const photo = photos[index];

  const onViewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems[0]?.index != null) setIndex(viewableItems[0].index);
  }).current;

  const share = useCallback(async () => {
    setSharing(true);
    try {
      await Sharing.shareAsync(await exportForSharing(photo.uri));
    } catch {
      Alert.alert("Couldn't share this photo", "If it's stored in iCloud, open it in Photos first.");
    } finally {
      setSharing(false);
    }
  }, [photo]);

  const renderItem = useCallback(
    ({ item }) => <ZoomablePhoto photo={item} width={width} height={height} />,
    [height, width]
  );

  if (!photo) return null;

  return (
    <Modal visible animationType="fade" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.container}>
        <FlatList
          data={photos}
          horizontal
          pagingEnabled
          initialScrollIndex={startIndex}
          getItemLayout={(_, i) => ({ length: width, offset: width * i, index: i })}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          onViewableItemsChanged={onViewableItemsChanged}
          viewabilityConfig={{ itemVisiblePercentThreshold: 60 }}
          showsHorizontalScrollIndicator={false}
          windowSize={3}
          initialNumToRender={1}
          maxToRenderPerBatch={2}
        />
        <View style={[styles.topBar, { paddingTop: insets.top + 6 }]}>
          <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button" accessibilityLabel="Close" style={styles.iconButton}>
            <Ionicons name="close" size={26} color="#fff" />
          </Pressable>
          <View style={styles.spacer} />
          <Pressable
            onPress={() => setShowInfo((value) => !value)}
            hitSlop={12}
            accessibilityRole="button"
            accessibilityLabel="Photo details"
            style={styles.iconButton}
          >
            <Ionicons name={showInfo ? "information-circle" : "information-circle-outline"} size={26} color="#fff" />
          </Pressable>
          <Pressable onPress={share} disabled={sharing} hitSlop={12} accessibilityRole="button" accessibilityLabel="Share" style={styles.iconButton}>
            {sharing ? <ActivityIndicator color="#fff" /> : <Ionicons name="share-outline" size={25} color="#fff" />}
          </Pressable>
        </View>
        {showInfo && (
          <View style={[styles.info, { paddingBottom: insets.bottom + 16 }]}>
            <Text numberOfLines={1} style={styles.infoTitle}>{photo.filename}</Text>
            <Text style={styles.infoText}>{formatDate(photo.creationTime)}</Text>
            <Text style={styles.infoText}>{photo.width} × {photo.height}</Text>
          </View>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  topBar: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingBottom: 6,
    backgroundColor: "rgba(0,0,0,0.35)",
  },
  spacer: { flex: 1 },
  iconButton: { width: 44, height: 44, alignItems: "center", justifyContent: "center" },
  info: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 20,
    paddingTop: 16,
    gap: 4,
    backgroundColor: "rgba(0,0,0,0.65)",
  },
  infoTitle: { color: "#fff", fontSize: 15, fontWeight: "600" },
  infoText: { color: "#D5D7DE", fontSize: 13.5 },
});
