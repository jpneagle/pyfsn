# Media Preview Feature

## Overview

pyfsn includes a media preview feature that displays thumbnails when hovering over image and video files in the 3D view. This feature enhances file browsing by providing visual previews without needing to open files.

## Features

### Image Preview
- **Automatic Detection**: Recognizes 15+ image formats
- **Scaled Thumbnails**: Maximum 320×240 pixels, preserving aspect ratio
- **Dimension Display**: Shows image size in the tooltip
- **Caching**: Pre-scales and caches images for performance

### Video Preview
- **Automatic Detection**: Recognizes 15+ video formats
- **Dynamic Scene Preview**: Cycles through 4 scenes (at 10%, 30%, 50%, 70% of duration), ~2 seconds each
- **Background Thread**: All video I/O runs on a `QThread` (`VideoPlayerThread`) to prevent UI freezes
- **Non-blocking Cleanup**: Detached threads are kept alive until they finish, preventing crashes on corrupt files
- **Resolution & Duration Display**: Shows video dimensions, FPS, and duration in the tooltip
- **Purple-tinted Background**: Visual distinction from image previews
- **Graceful Error Handling**: Corrupt or unreadable files display an error message without freezing

## Installation

### Basic (Images only)
```bash
pip install -e .
```

### With Video Support
```bash
pip install -e ".[video]"
```

Or manually:
```bash
pip install opencv-python
```

For headless systems, use the lighter version:
```bash
pip install opencv-python-headless
```

## Supported Formats

### Image Formats
| Extension | Format |
|-----------|--------|
| `.png` | PNG |
| `.jpg`, `.jpeg`, `.jpe`, `.jfif` | JPEG |
| `.gif` | GIF |
| `.bmp` | Bitmap |
| `.webp` | WebP |
| `.svg` | SVG |
| `.ico` | Icon |
| `.tiff`, `.tif` | TIFF |
| `.psd` | Photoshop |
| `.raw`, `.cr2`, `.nef`, `.arw` | Camera RAW |
| `.heic`, `.heif` | HEIF |
| `.avif` | AVIF |

### Video Formats
| Extension | Format |
|-----------|--------|
| `.mp4` | MP4 |
| `.avi` | AVI |
| `.mkv` | Matroska |
| `.mov` | QuickTime |
| `.wmv` | Windows Media |
| `.flv` | Flash Video |
| `.webm` | WebM |
| `.m4v` | M4V |
| `.mpg`, `.mpeg` | MPEG |
| `.3gp` | 3GP |
| `.ogv` | OGG Video |
| `.m2ts`, `.mts` | MPEG Transport Stream |
| `.vob` | DVD Video |
| `.rm`, `.rmvb` | RealMedia |
| `.asf` | ASF |
| `.divx`, `.xvid` | DivX/Xvid |
| `.f4v` | F4V |
| `.mxf` | MXF |
| `.qt` | QuickTime |

## API Usage

### Node Detection

```python
from pyfsn.model import Node
from pathlib import Path

# Check if a file is an image
image_node = Node.from_path(Path("photo.jpg"))
print(image_node.is_image_file)  # True
print(image_node.is_video_file)  # False

# Check if a file is a video
video_node = Node.from_path(Path("movie.mp4"))
print(video_node.is_image_file)  # False
print(video_node.is_video_file)  # True
```

### Tooltip Integration

```python
from pyfsn.view.main_window import ImagePreviewTooltip

# The tooltip is automatically created by MainWindow
# Access it via:
tooltip = window.file_tooltip

# Show tooltip for a node
tooltip.show_for_node(node, x=100, y=200)

# Hide tooltip
tooltip.hide()

# Clear cache (useful when files change)
tooltip.clear_cache()
```

## Implementation Details

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     MainWindow                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │              ImagePreviewTooltip                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  File Detection (Node.is_image/video_file)   │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Image Loading (QPixmap - main thread)       │  │  │
│  │  │  - _load_image_if_needed()                   │  │  │
│  │  │  - Pre-scaling on load                       │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Video Preview (Background QThread)          │  │  │
│  │  │  - VideoPlayerThread runs cv2 operations     │  │  │
│  │  │  - Emits frame_ready / info_ready / error    │  │  │
│  │  │  - 4-scene digest playback                   │  │  │
│  │  │  - Non-blocking stop & detached cleanup      │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Rendering (paintEvent)                      │  │  │
│  │  │  - Text info (name, size, permissions)       │  │  │
│  │  │  - Preview image/video with aspect ratio     │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Performance Considerations

1. **Caching Strategy**
   - Images are loaded and scaled once, then cached
   - Video frames are streamed from the background thread
   - Cache is keyed by file path

2. **Pre-scaling**
   - Images are pre-scaled to max 320×240 on load
   - Video frames are scaled on receipt from thread
   - Aspect ratio is preserved

3. **Thread Safety**
   - Video I/O runs entirely on `VideoPlayerThread` (QThread)
   - Communication via Qt signals (`frame_ready`, `info_ready`, `error_occurred`)
   - Non-blocking stop: `stop()` sets a flag but does not `wait()`
   - Detached threads are held in `_detached_threads` list to prevent GC crashes
   - Signals are disconnected before stopping to prevent stale updates

4. **Memory Management**
   - Old cache is cleared when switching files
   - `clear_cache()` method for manual cleanup
   - Detached threads auto-cleanup via `finished` signal + `deleteLater()`

### Error Handling

| Scenario | Behavior |
|----------|----------|
| OpenCV not installed | Shows "Video (OpenCV not available)" |
| Unreadable video file | Shows "Video unreadable" |
| Corrupt video (e.g. truncated MKV) | Thread detaches gracefully, UI stays responsive |
| Corrupted image | Shows file info without preview |
| File not found | Tooltip doesn't appear |
| Thread stuck in cv2 | Thread is detached; cleaned up when it finishes |

## Troubleshooting

### Video thumbnails not showing

**Problem**: Video files show "Video (OpenCV not available)"

**Solution**:
```bash
pip install opencv-python
```

### Large images causing lag

**Problem**: Large image files cause tooltip to lag

**Solution**: Images are pre-scaled to 320×240 maximum. If still lagging:
- Check available memory
- Clear cache with `tooltip.clear_cache()`

### Permission errors

**Problem**: Can't read media files

**Solution**: Ensure file permissions allow reading:
```bash
chmod +r /path/to/file
```

## Future Enhancements

- [x] Video duration display
- [x] Dynamic video scene preview
- [x] Background thread for video I/O
- [ ] GIF animation support
- [ ] Thumbnail size configuration
- [ ] EXIF data display for images
- [ ] Codec information for videos
