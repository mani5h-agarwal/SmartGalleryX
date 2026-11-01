import { File } from "expo-file-system";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import {
  Asset,
  AssetField,
  MediaType,
  Query,
  getPermissionsAsync,
  requestPermissionsAsync,
} from "expo-media-library";
import { Platform } from "react-native";

// Matches the model's input. It squashes every photo to this square, so sending
// anything larger would only cost upload time.
const MODEL_IMAGE_SIZE = 384;
const THUMBNAIL_QUALITY = 0.8;
// Decoding bound. A 48 MP photo needs about 200 MB of memory in full but 12 MB at this
// size, and it stays well above the model's input even for wide panoramas.
const MAX_DECODE_SIZE = 1024;
const SHARE_QUALITY = 0.95;

const PHOTO_PERMISSION = ["photo"];

// iOS asset ids are ph:// addresses, and native image loaders fetch those at the size
// they need. Asset URIs there point at the full-size file, and getting one can prepare
// the photo for editing or download it from iCloud.
const USE_ID_AS_URI = Platform.OS === "ios";

export const getPhotoPermission = () => getPermissionsAsync(false, PHOTO_PERMISSION);
export const requestPhotoPermission = () => requestPermissionsAsync(false, PHOTO_PERMISSION);

const newestFirst = () =>
  new Query()
    .eq(AssetField.MEDIA_TYPE, MediaType.IMAGE)
    .orderBy({ key: AssetField.CREATION_TIME, ascending: false });

// A light read for syncing: no URIs, so scanning a large library stays fast.
export async function readLibraryPage(offset, limit) {
  const page = await newestFirst().limit(limit).offset(offset).exeForMetadata();
  return page.map((photo) => ({
    id: photo.id,
    modifiedAt: photo.modificationTime ?? photo.creationTime ?? 0,
  }));
}

export async function readGalleryPage(offset, limit) {
  const query = newestFirst().limit(limit).offset(offset);
  if (USE_ID_AS_URI) {
    const page = await query.exeForMetadata();
    return page.map((photo) => ({ ...photo, uri: photo.id }));
  }
  const assets = await query.exe();
  return Promise.all(assets.map((asset) => asset.getInfo()));
}

async function readPhotoById(id) {
  const asset = new Asset(id);
  if (!USE_ID_AS_URI) return asset.getInfo();
  const [filename, width, height, creationTime] = await Promise.all([
    asset.getFilename(),
    asset.getWidth(),
    asset.getHeight(),
    asset.getCreationTime(),
  ]);
  return { id, uri: id, filename, width, height, creationTime };
}

// Photos deleted since they were indexed are dropped, not shown as blanks.
export async function readPhotosById(ids) {
  const photos = await Promise.all(
    ids.map((id) => readPhotoById(id).catch(() => null))
  );
  return photos.filter(Boolean);
}

async function render(uri, resize, compress, loadOptions) {
  const context = ImageManipulator.manipulate(uri, loadOptions);
  if (resize) context.resize(resize);
  const image = await context.renderAsync();
  try {
    const result = await image.saveAsync({ format: SaveFormat.JPEG, compress });
    return result.uri;
  } finally {
    image.release();
    context.release();
  }
}

export const thumbnails = {
  async create(id) {
    const uri = USE_ID_AS_URI ? id : await new Asset(id).getUri();
    return render(uri, { width: MODEL_IMAGE_SIZE, height: MODEL_IMAGE_SIZE }, THUMBNAIL_QUALITY, {
      maxWidth: MAX_DECODE_SIZE,
      maxHeight: MAX_DECODE_SIZE,
    });
  },
  async release(uris) {
    for (const uri of uris) {
      try {
        new File(uri).delete();
      } catch {
        // The OS clears this cache folder anyway.
      }
    }
  },
};

export const exportForSharing = (uri) => render(uri, null, SHARE_QUALITY);
