"""Sound effects for pyfsn 3D file system visualization.

Generates simple UI sounds using numpy sine waves. No external audio files needed.
Uses QSoundEffect from PyQt6.QtMultimedia for playback.
"""

import io
import struct
import numpy as np

# Sound playback is optional - gracefully handle missing QtMultimedia
_SOUND_AVAILABLE = False
try:
    from PyQt6.QtMultimedia import QSoundEffect
    from PyQt6.QtCore import QUrl, QTemporaryFile, QBuffer, QIODevice
    _SOUND_AVAILABLE = True
except ImportError:
    pass


def _generate_wav_bytes(samples: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Generate WAV file bytes from numpy samples.

    Args:
        samples: Audio samples as float32 array (-1.0 to 1.0)
        sample_rate: Sample rate in Hz

    Returns:
        WAV file as bytes
    """
    # Convert to 16-bit PCM
    pcm = (samples * 32767).astype(np.int16)
    raw = pcm.tobytes()

    # Build WAV header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(raw)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )

    return header + raw


def _make_click_sound() -> bytes:
    """Generate a short click sound (50ms, 800Hz)."""
    sr = 44100
    duration = 0.05
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # 800Hz sine with fast exponential decay
    envelope = np.exp(-t * 60)
    samples = np.sin(2 * np.pi * 800 * t) * envelope * 0.4
    return _generate_wav_bytes(samples, sr)


def _make_navigate_sound() -> bytes:
    """Generate a navigation sweep sound (200ms, 300-600Hz)."""
    sr = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Frequency sweep from 300 to 600Hz
    freq = 300 + 300 * (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    envelope = np.sin(np.pi * t / duration)  # Smooth fade in/out
    samples = np.sin(phase) * envelope * 0.3
    return _generate_wav_bytes(samples, sr)


def _make_error_sound() -> bytes:
    """Generate an error buzz sound (150ms, 200Hz)."""
    sr = 44100
    duration = 0.15
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    envelope = np.exp(-t * 15)
    samples = np.sin(2 * np.pi * 200 * t) * envelope * 0.35
    return _generate_wav_bytes(samples, sr)


class SoundManager:
    """Manages UI sound effects for the file system navigator.

    Sounds are generated procedurally using numpy. Disabled by default.
    Requires PyQt6.QtMultimedia to be available.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._sounds: dict[str, object] = {}
        self._temp_files: list = []  # Keep references to prevent GC

        if _SOUND_AVAILABLE:
            self._init_sounds()

    def _init_sounds(self) -> None:
        """Initialize sound effects from generated WAV data."""
        sound_data = {
            'click': _make_click_sound(),
            'navigate': _make_navigate_sound(),
            'error': _make_error_sound(),
        }

        import tempfile
        import os

        for name, wav_bytes in sound_data.items():
            try:
                # Write WAV to a temporary file (QSoundEffect needs a file URL)
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp.write(wav_bytes)
                tmp.flush()
                tmp.close()
                self._temp_files.append(tmp.name)

                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(tmp.name))
                effect.setVolume(0.5)
                self._sounds[name] = effect
            except Exception:
                pass  # Silently skip if sound init fails

    @property
    def enabled(self) -> bool:
        """Whether sound effects are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable sound effects."""
        self._enabled = value

    def toggle(self) -> bool:
        """Toggle sound effects on/off. Returns new state."""
        self._enabled = not self._enabled
        return self._enabled

    def play_click(self) -> None:
        """Play click sound (for selection)."""
        self._play('click')

    def play_navigate(self) -> None:
        """Play navigation sound (for directory traversal)."""
        self._play('navigate')

    def play_error(self) -> None:
        """Play error sound."""
        self._play('error')

    def _play(self, name: str) -> None:
        """Play a named sound if enabled and available."""
        if not self._enabled or not _SOUND_AVAILABLE:
            return
        effect = self._sounds.get(name)
        if effect:
            effect.play()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import os
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()
