"""The charging animation, packed as the charging module (a directory).

When the phone is plugged in on the lock screen, SystemUI (ChargingAnimController.startThemeChargeAnim) hands over to
com.hihonor.livewallpaper.paradise, which dims the screen with <background_color>, plays charging_video.mp4 on top and
draws the lightning icon, the battery level and the charging prompt at the spots charging_config.xml gives, in a
1080 x 2160 frame (<base_scale>). The video plays frames <frame_init_begin>..<frame_init_end> once, then loops
<frame_loop_begin>..<frame_loop_end>.

The video is H.264 with its transparency in a third track, as in Honor's official theme 荣耀机器人: a data track
(handler MINE, sample entry mp4s, udta name HonorPrivateData) with one sample per video frame. Each sample is an LZ4
frame of an 8-bit alpha mask at half size, 540 x 1080 in rows of 544 bytes (half the 1088px coded width). The first
sample starts with the 24-byte header b'HonorPrivateData' 00 01 00 00 00 00 00 00. Color is not premultiplied.

Here the XANA symbol (as on the fingerprint icon, see fingerprint.py) turns red from the bottom up, then glows and
sends out the rings of a tower's aura, looping every three seconds.
"""

import hashlib
import io
import os
import shutil
import struct
import tempfile
from importlib.metadata import version
from pathlib import Path

import av
import lz4.frame
import numpy as np
import numpy.typing as npt
import resvg_py
from av.video.reformatter import ColorRange, Colorspace
from PIL import Image, ImageDraw

from . import fingerprint, previews
from .fingerprint import RED, WHITE, blur, ease
from .paths import BUILD, ROOT

type Mask = npt.NDArray[np.float64]  # alpha 0..1

W, H = 1080, 2160  # <base_scale>
FPS = 30
RATE = 44100  # the silent sound track's sample rate
INIT, LOOP = 30, 90  # frames turning red, frames of the loop
FRAMES = INIT + LOOP
SIZE = 520  # px, the symbol's height
TOP, BOTTOM = 280, 1480  # rows the animation can reach; the battery level and prompt sit below
CY = 860  # the symbol's center
GOP = 30  # a key frame every second, so the loop starts on one
MAGIC = b'HonorPrivateData' + bytes([0, 1, 0, 0, 0, 0, 0, 0])
STRIDE = 544  # bytes per alpha row
CACHE = BUILD / 'charging/cache'
NAMES = ('charging_config.xml', 'contentLang.xml', 'lightning.png', 'charging_video.mp4')
# what the module depends on: this code, the XANA symbol, the review GIF's wallpaper and fonts, and the packages that
# draw and encode it (their versions decide the video's bytes)
INPUTS = (
    'theme/charging.py',
    'theme/fingerprint.py',
    'theme/previews.py',
    'theme/paths.py',
    'data/fingerprint',
    'data/inter',
)
PACKAGES = ('av', 'lz4', 'numpy', 'pillow', 'resvg-py')
BOLT = (26, 67)  # the lightning icon, as in 荣耀机器人
# The player puts % GAP px after the level only while a centered "90.01" can't overlap a % centered on x. When they
# overlap, it slides % left half a digit per missing char, which lands it on "72". x="900" keeps them apart.
GAP = 12  # px between the battery level and %
CONFIG = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ChargingConfig>
  <version>2.0.0</version>
  <base_scale width="{W}" height="{H}"/>
  <video_framerate framerate="{FPS}"/>
  <background_color>#D9000000</background_color>
  <lightning_icon x="540" y="1687" w="104" h="104" path="./lightning.png"/>
  <progress_text content="72.70" x="540" y="1836" size="122" color="#FFFFFFFF" path=""/>
  <percentage_text content="%" x="900" y="1858" x_rel="{GAP}" size="50" color="#ffffffff" path=""/>
  <charging_prompt_texts>
    <level0 content="Fast charging" x="540" y="1945" size="45" color="#87FFFFFF" path=""/>
    <level1 content="SuperCharge" x="540" y="1945" size="45" color="#87FFFFFF" path=""/>
    <level2 content="SuperCharge" x="540" y="1945" size="45" color="#87FFFFFF" path=""/>
  </charging_prompt_texts>
  <video_segment path="./" file="charging_video.mp4" alias="">
    <clip_item tag="xana" screen="0x03" orientation="0x00">
      <frame_init_begin>0</frame_init_begin>
      <frame_init_end>{INIT}</frame_init_end>
      <frame_loop_begin>{INIT}</frame_loop_begin>
      <frame_loop_end>{FRAMES}</frame_loop_end>
    </clip_item>
  </video_segment>
</ChargingConfig>
"""
LANG = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<resources>
  <string name="level0" lang="en">Fast charging</string>
  <string name="level0" lang="zh">快速充电</string>
  <string name="level1" lang="en">SuperCharge</string>
  <string name="level1" lang="zh">超级充电</string>
  <string name="level2" lang="en">SuperCharge</string>
  <string name="level2" lang="zh">超级充电</string>
</resources>
"""


def frames() -> list[Image.Image]:
    """Every frame as RGBA, cropped to rows TOP..BOTTOM."""
    h = BOTTOM - TOP
    png = resvg_py.svg_to_bytes(svg_string=fingerprint.svg(), height=SIZE)
    glyph = np.zeros((h, W))
    x0, y0 = (W - SIZE) // 2, CY - TOP - SIZE // 2
    glyph[y0 : y0 + SIZE, x0 : x0 + SIZE] = np.asarray(Image.open(io.BytesIO(bytes(png))).getchannel('A')) / 255
    near, far = blur(glyph, 0.04 * SIZE), blur(glyph, 0.18 * SIZE)
    y, x = np.mgrid[:h, :W] + 0.5
    r = np.hypot(x - W / 2, y - (CY - TOP))
    soft = 0.15 * SIZE  # the red front's height
    out = []
    for i in range(FRAMES):
        # the red front climbs from below the symbol to above it, as on the fingerprint
        front = y0 + SIZE + soft - ease(i / INIT) * (SIZE + 2 * soft)
        red = ease((y - front) / soft + 0.5)[..., None]
        color = WHITE * (1 - red) + RED * red
        # t runs 0..1 over the loop; the glow breathes once per loop and two aura rings leave the symbol half a loop
        # apart, so the last frame flows into the first. During the turn the rings fade in.
        t = (i - INIT) / LOOP
        glow = ease(i / INIT) * (1 + 0.2 * np.sin(2 * np.pi * t))
        ring = np.zeros_like(r)
        for k in (0, 0.5):
            s = (t + k) % 1
            ring += (1 - s) ** 1.5 * np.exp(-(((r - (0.45 + 0.6 * s) * SIZE) / (0.035 * SIZE)) ** 2))
        ring *= ease(i / INIT)
        halo = np.clip(glow * (0.8 * near + 0.45 * far) + 0.5 * ring, 0, 1)
        a = glyph + halo * (1 - glyph)
        rgb = (color * glyph[..., None] + RED * (halo * (1 - glyph))[..., None]) / np.maximum(a, 1e-6)[..., None]
        im = np.dstack([np.clip(rgb, 0, 255) * (a > 0)[..., None], a * 255])
        out.append(Image.fromarray(np.round(im).astype(np.uint8), 'RGBA'))
    return out


def bolt() -> Image.Image:
    """lightning.png: a slanted bolt in the tower's red."""
    s = 8
    w, h = BOLT
    pts = [(26, 0), (4, 39), (13.5, 39), (0, 67), (23, 27), (13, 27), (26, 0)]
    im = Image.new('L', (w * s, h * s))
    ImageDraw.Draw(im).polygon([(px * s, py * s) for px, py in pts], fill=255)
    a = im.resize(BOLT, Image.Resampling.LANCZOS)
    out = Image.new('RGBA', BOLT, tuple(int(c) for c in RED) + (0,))
    out.putalpha(a)
    return out


def mockup(fs: list[Image.Image], dst: Path) -> None:
    """An animated GIF of the lock screen while charging: the turn, then the loop twice, at a third of the size."""
    bg = previews.view(W, H)
    bg.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, 0xD9)))
    b = bolt()
    bg.alpha_composite(b, (540 - b.width // 2, 1687 - b.height // 2))
    d = ImageDraw.Draw(bg)
    d.text((540, 1836), '72', fill='white', font=previews.font(122, True), anchor='mm')
    # x_rel: like the player, % starts GAP px after the level's advance width, not its ink
    right = 540 + d.textlength('72', font=previews.font(122, True)) / 2
    d.text((right + GAP, 1858), '%', fill='white', font=previews.font(50, True), anchor='lm')
    d.text((540, 1945), 'Fast charging', fill=(255, 255, 255, 0x87), font=previews.font(45), anchor='mm')
    shots = []
    for f in fs + fs[INIT:]:
        im = bg.copy()
        im.alpha_composite(f, (0, TOP))
        shots.append(im.convert('RGB').resize((W // 3, H // 3), Image.Resampling.LANCZOS))
    shots[0].save(dst, save_all=True, append_images=shots[1:], duration=1000 // FPS, loop=0)


def video(fs: list[Image.Image]) -> bytes:
    """H.264 of the color (black where transparent), muxed by PyAV, plus the alpha track."""
    buf = io.BytesIO()
    with av.open(buf, 'w', format='mp4') as out:
        s = out.add_stream('libx264', rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, 'yuv420p'
        # baseline like the official video; one thread so every machine encodes the same bytes
        s.options = {'profile': 'baseline', 'crf': '18', 'g': str(GOP), 'keyint_min': str(GOP), 'threads': '1'}
        s.thread_count = 1
        # tagged BT.709 limited range, so the decoder doesn't guess the matrix and shift the red
        cc = s.codec_context
        cc.colorspace = cc.color_primaries = cc.color_trc = 1
        cc.color_range = ColorRange.MPEG
        # the official video's tracks are video, audio, alpha; keep that order with a silent track
        sound = out.add_stream('aac', rate=RATE, layout='stereo')
        for f in fs:
            canvas = Image.new('RGB', (W, H))
            canvas.paste(f.convert('RGB'), (0, TOP))
            frame = av.VideoFrame.from_image(canvas).reformat(
                format='yuv420p', dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG
            )
            out.mux(s.encode(frame))
        out.mux(s.encode())
        silence = np.zeros((1, 2 * 1024), np.float32)
        for k in range(FRAMES * RATE // FPS // 1024 + 1):
            frame = av.AudioFrame.from_ndarray(silence, format='flt', layout='stereo')
            frame.sample_rate, frame.pts = RATE, k * 1024
            out.mux(sound.encode(frame))
        out.mux(sound.encode())
    return with_alpha(buf.getvalue(), [alpha(f) for f in fs])


def alpha(f: Image.Image) -> bytes:
    """One sample of the alpha track: the frame's alpha at half size, LZ4 compressed."""
    a = np.zeros((H // 2, STRIDE), np.uint8)
    half = f.getchannel('A').resize((W // 2, f.height // 2), Image.Resampling.BOX)
    a[TOP // 2 : TOP // 2 + half.height, : W // 2] = np.asarray(half)
    return lz4.frame.compress(
        a.tobytes(),
        block_size=lz4.frame.BLOCKSIZE_MAX64KB,
        block_linked=True,
        content_checksum=False,
        store_size=False,
    )


def box(kind: bytes, *parts: bytes) -> bytes:
    body = b''.join(parts)
    return struct.pack('>I4s', 8 + len(body), kind) + body


def full(kind: bytes, version_flags: int, *parts: bytes) -> bytes:
    return box(kind, struct.pack('>I', version_flags), *parts)


def descriptor(tag: int, body: bytes) -> bytes:
    """An MPEG-4 descriptor with the 4-byte length the official video uses."""
    n = len(body)
    return bytes([tag, 0x80 | n >> 21 & 0x7F, 0x80 | n >> 14 & 0x7F, 0x80 | n >> 7 & 0x7F, n & 0x7F]) + body


def children(data: bytes) -> list[tuple[bytes, bytes]]:
    out, o = [], 0
    while o < len(data):
        size, kind = struct.unpack('>I4s', data[o : o + 8])
        out.append((kind, data[o : o + size]))
        o += size
    return out


def with_alpha(mp4: bytes, samples: list[bytes]) -> bytes:
    """Add the HonorPrivateData track to PyAV's mp4 (ftyp, free, mdat, moov), its samples in an mdat after moov."""
    samples = [MAGIC + samples[0], *samples[1:]]
    top = children(mp4)
    assert [k for k, _ in top][-1] == b'moov', [k for k, _ in top]
    head = b''.join(b for k, b in top[:-1])
    moov = children(top[-1][1][8:])
    mvhd = bytearray(dict(moov)[b'mvhd'])
    # version 0 mvhd: duration (in its timescale, which tkhd shares) at 24, next track ID in the last 4 bytes
    duration = struct.unpack('>I', mvhd[24:28])[0]
    track = struct.unpack('>I', mvhd[-4:])[0]
    mvhd[-4:] = struct.pack('>I', track + 1)
    n, delta, ts = len(samples), 20, 600  # as in the official video: 600 units a second
    sizes = [len(s) for s in samples]
    esds = full(
        b'esds',
        0,
        descriptor(
            3,
            struct.pack('>HB', 0, 0)
            + descriptor(4, struct.pack('>BB', 1, 0x81) + max(sizes).to_bytes(3) + struct.pack('>II', 0, 0))
            + descriptor(6, b'\x02'),
        ),
    )
    entry = box(b'mp4s', bytes(6), struct.pack('>H', 1), esds)

    def trak(chunk: int) -> bytes:
        tkhd = full(
            b'tkhd',
            1,  # enabled
            struct.pack('>IIIII', 0, 0, track, 0, duration),
            bytes(8),
            struct.pack('>hhhH', 0, 0, 0, 0),
            struct.pack('>9I', 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000),
            struct.pack('>II', STRIDE * 2 << 16, H << 16),
        )
        mdhd = full(b'mdhd', 0, struct.pack('>IIIIHH', 0, 0, ts, n * delta, 0x55C4, 0))
        hdlr = full(b'hdlr', 0, struct.pack('>I4s', 0, b'MINE'), bytes(12), b'\0')
        dinf = box(b'dinf', full(b'dref', 0, struct.pack('>I', 1), full(b'url ', 1)))
        stbl = box(
            b'stbl',
            full(b'stsd', 0, struct.pack('>I', 1), entry),
            full(b'stts', 0, struct.pack('>III', 1, n, delta)),
            full(b'stsz', 0, struct.pack('>II', 0, n), struct.pack(f'>{n}I', *sizes)),
            full(b'stsc', 0, struct.pack('>IIII', 1, 1, n, 1)),
            full(b'stco', 0, struct.pack('>II', 1, chunk)),
        )
        minf = box(b'minf', full(b'nmhd', 0), dinf, stbl)
        udta = box(b'udta', box(b'name', b'HonorPrivateData'))
        return box(b'trak', tkhd, box(b'mdia', mdhd, hdlr, minf), udta)

    body = [bytes(mvhd) if k == b'mvhd' else b for k, b in moov]
    size = len(box(b'moov', *body, trak(0)))
    new = box(b'moov', *body, trak(len(head) + size + 8))
    assert len(new) == size
    return head + new + box(b'mdat', *samples)


def key() -> str:
    h = hashlib.sha256()
    for p in PACKAGES:
        h.update(f'{p}=={version(p)}\0'.encode())
    for f in sorted(f for i in INPUTS for f in [ROOT / i, *(ROOT / i).rglob('*')] if f.is_file()):
        h.update(str(f.relative_to(ROOT)).encode() + b'\0' + f.read_bytes())
    return h.hexdigest()


def files() -> dict[str, bytes]:
    """The module's files. Rendering them takes most of a build, so this keeps them in build/charging/cache and renders
    them again only when one of the INPUTS or PACKAGES changes. A render outside CI also writes a GIF of the
    animation to build/charging for review."""
    k = key()
    try:
        return {n: (CACHE / k / n).read_bytes() for n in NAMES}
    except FileNotFoundError:
        pass
    fs = frames()
    review = BUILD / 'charging'
    review.mkdir(parents=True, exist_ok=True)
    # nobody looks at the GIF on CI, and it takes a third of the render
    if not os.environ.get('CI'):
        mockup(fs, review / 'mockup.gif')
    png = io.BytesIO()
    bolt().save(png, 'PNG')
    out: dict[str, bytes] = dict(zip(NAMES, (CONFIG.encode(), LANG.encode(), png.getvalue(), video(fs)), strict=True))
    # write to a new directory and rename it into place, so a build never reads a half-written cache
    CACHE.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.', dir=CACHE))
    for name, data in out.items():
        (tmp / name).write_bytes(data)
    shutil.rmtree(CACHE / k, ignore_errors=True)  # drop a cache that lost some of its files
    try:
        tmp.rename(CACHE / k)
    except OSError:  # another build cached the same key first
        shutil.rmtree(tmp)
    for old in CACHE.iterdir():
        if old.name != k and not old.name.startswith('.'):
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    return out


def build(outer: Path, files: dict[str, bytes]) -> None:
    mod = outer / 'charging'
    mod.mkdir(parents=True)
    for name, data in files.items():
        (mod / name).write_bytes(data)


if __name__ == '__main__':
    shutil.rmtree(BUILD / 'charging/module', ignore_errors=True)
    build(BUILD / 'charging/module', files())
