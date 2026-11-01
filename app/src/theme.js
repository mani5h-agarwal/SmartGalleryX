import { useColorScheme } from "react-native";

const light = {
  dark: false,
  background: "#FFFFFF",
  surface: "#F3F4F7",
  surfaceStrong: "#E7E9EF",
  text: "#14161C",
  textMuted: "#646A78",
  accent: "#4257E8",
  accentSoft: "#E8EBFE",
  onAccent: "#FFFFFF",
  danger: "#C5362B",
  dangerSoft: "#FCEBE9",
  success: "#1E8A4C",
  separator: "#E7E9EF",
};

const dark = {
  dark: true,
  background: "#0B0D12",
  surface: "#191C24",
  surfaceStrong: "#262A35",
  text: "#F1F2F6",
  textMuted: "#9AA0AF",
  accent: "#7E90FF",
  accentSoft: "#232A52",
  onAccent: "#0B0D12",
  danger: "#FF8A80",
  dangerSoft: "#3A1D1B",
  success: "#5FD08E",
  separator: "#262A35",
};

export const useTheme = () => (useColorScheme() === "dark" ? dark : light);
