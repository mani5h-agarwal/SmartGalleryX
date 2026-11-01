import { StatusBar } from "expo-status-bar";
import { Image } from "expo-image";
import * as SystemUI from "expo-system-ui";
import { useEffect } from "react";
import { SafeAreaProvider } from "react-native-safe-area-context";

import GalleryScreen from "./src/screens/GalleryScreen";
import { GalleryProvider } from "./src/state/GalleryContext";
import { useTheme } from "./src/theme";

// The default memory cache has no limit, and a long scroll through a large library fills it.
Image.configureCache({ maxMemoryCost: 150 * 1024 * 1024, maxMemoryCount: 400 });

export default function App() {
  const theme = useTheme();

  useEffect(() => {
    SystemUI.setBackgroundColorAsync(theme.background);
  }, [theme.background]);

  return (
    <SafeAreaProvider>
      <GalleryProvider>
        <StatusBar style="auto" />
        <GalleryScreen />
      </GalleryProvider>
    </SafeAreaProvider>
  );
}
