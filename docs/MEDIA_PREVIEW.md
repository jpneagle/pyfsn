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
- **Smart Thumbnail Extraction**: Extracts frame at 25% of video duration for representative preview
- **Play Icon Overlay**: Visual indicator distinguishing videos from images
- **Resolution Display**: Shows video dimensions in the tooltip
- **Purple-tinted Background**: Visual distinction from image previews

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
| `.ts`, `.m2ts`, `.mts` | MPEG Transport Stream |
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
┌─────────────────────────────────────────────────────────┐
│                    MainWindow                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │              ImagePreviewTooltip                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  File Detection (Node.is_image_file)        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Image Loading (QPixmap)                    │  │  │
│  │  │  - _load_image_if_needed()                  │  │  │
│  │  │  - Pre-scaling on load                      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Video Loading (OpenCV - optional)          │  │  │
│  │  │  - _load_video_thumbnail_if_needed()        │  │  │
│  │  │  - Frame extraction at 25%                  │  │  │
│  │  │  - Graceful fallback without OpenCV         │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Rendering (paintEvent)                     │  │  │
│  │  │  - Text info (name, size, permissions)     │  │  │
│  │  │  - Preview image with aspect ratio          │  │  │
│  │  │  - Play icon for videos                     │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Performance Considerations

1. **Caching Strategy**
   - Images are loaded and scaled once, then cached
   - Video thumbnails are extracted once per session
   - Cache is keyed by file path

2. **Pre-scaling**
   - Images are pre-scaled to max 320×240 on load
   - Reduces CPU usage during rendering
   - Aspect ratio is preserved

3. **Memory Management**
   - Old cache is cleared when switching files
   - `clear_cache()` method for manual cleanup
   - QPixmaps are managed by Qt's memory system

### Error Handling

| Scenario | Behavior |
|----------|----------|
| OpenCV not installed | Shows "Video (OpenCV not available)" |
| Unreadable video file | Shows "Video (unreadable)" |
| Corrupted image | Shows file info without preview |
| File not found | Tooltip doesn't appear |

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

- [ ] GIF animation support
- [ ] Video duration display
- [ ] Thumbnail size configuration
- [ ] Multiple thumbnail pages for videos
- [ ] EXIF data display for images
- [ ] Codec information for videos
