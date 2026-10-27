"""Status bar icons, packed as the com.android.systemui.status module.

The module is a zip of res/drawable-xxhdpi/<name>.png; each PNG replaces the SystemUI drawable of that name. The
names follow Honor's official theme 荣耀机器人. Every canvas has the size of the stock drawable's viewport
(data/statusbar/sizes.tsv): SystemUI stretches a theme PNG to the stock drawable's bounds. The layout inside each canvas
follows the stock SystemUI vectors, since several drawables are stacked in one view (signal bars + network type + in/out
arrows, battery frame + level + charge bolt). SystemUI tints the glyphs, so they are black on transparent; only the low
battery levels keep their own color. Glyphs are Material Symbols Rounded (Apache 2.0, data/statusbar), text is Inter
Display Bold (OFL, data/inter).

The icons go in com.android.systemui.status, as in 荣耀机器人, which the Themes app merges into com.android.systemui
when the theme is applied. Shipped directly in com.android.systemui, every icon showed except the battery: SystemUI
kept the stock battery whatever the theme had (MagicOS 11). com.android.systemui itself carries theme.xml and an
empty night/ entry: without that entry SystemUI takes the theme as light-only
(FullScreenAodNotificationControllerImpl.checkTheme) and draws notifications, dialogs and the screenshot UI light in
dark mode. On SDK 37 the merge keeps it.
"""

import io
import zipfile
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import resvg_py
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .paths import BUILD, DATA

type Mask = npt.NDArray[np.float64]  # alpha 0..1 at S x the PNG size

S = 4  # supersampling
H = 60  # every icon is 60px tall (20dp at xxhdpi)
DIM = 0.3  # the unlit part of a meter, as in the stock vectors (fillAlpha 0.3)
LOW, CRITICAL = (0xFF, 0x9B, 0x1A), (0xF5, 0x48, 0x42)  # SystemUI's low battery colors (<= 20%, <= 10%)
MODULES = ('com.android.systemui', 'com.android.systemui.status')  # theme.xml + night/, the icons
# name -> (w, h) of the stock drawable
SIZES = {
    n: (float(w), float(h))
    for n, w, h in (line.split('\t') for line in (DATA / 'statusbar/sizes.tsv').read_text().splitlines())
}


def blank(w: float, h: float = H) -> Mask:
    return np.zeros((round(h * S), round(w * S)))


def draw(m: Mask, fn: Callable[[ImageDraw.ImageDraw], None]) -> Mask:
    """Run fn on an S-scaled drawing surface and return what it drew as a mask."""
    im = Image.new('L', (m.shape[1], m.shape[0]))
    fn(ImageDraw.Draw(im))
    return np.asarray(im) / 255


def rrect(m: Mask, box: tuple[float, float, float, float], r: float) -> Mask:
    x0, y0, x1, y1 = (v * S for v in box)
    return draw(m, lambda d: d.rounded_rectangle((x0, y0, x1, y1), r * S, fill=255))


def over(*layers: tuple[Mask, float]) -> Mask:
    """Stack (mask, alpha) layers; a later layer replaces what is under it."""
    out = np.zeros_like(layers[0][0])
    for m, a in layers:
        out = out * (1 - m) + m * a
    return out


@cache
def _symbol(name: str, em: float) -> Mask:
    """A Material symbol rendered with its 24-unit box em px tall, cropped to the glyph."""
    png = resvg_py.svg_to_bytes(svg_path=str(DATA / f'statusbar/{name}.svg'), height=round(em * S))
    a = Image.open(io.BytesIO(bytes(png))).getchannel('A')
    return np.asarray(a.crop(a.getbbox())) / 255


def symbol(name: str, h: float | None = None, w: float | None = None) -> Mask:
    """A Material symbol scaled so the glyph is h px tall, or w px wide."""
    g = _symbol(name, 96)
    k = h / (g.shape[0] / S) if h else w / (g.shape[1] / S)
    return _symbol(name, 96 * k)


def paste(m: Mask, g: Mask, cx: float, cy: float) -> Mask:
    """Put glyph g centered on (cx, cy) in a canvas the size of m."""
    out = np.zeros_like(m)
    y, x = round(cy * S - g.shape[0] / 2), round(cx * S - g.shape[1] / 2)
    if y < 0 or x < 0 or y + g.shape[0] > m.shape[0] or x + g.shape[1] > m.shape[1]:
        raise ValueError(f'glyph {g.shape} at ({cx}, {cy}) does not fit {m.shape}')
    out[y : y + g.shape[0], x : x + g.shape[1]] = g
    return out


def png(m: Mask, rgb: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    h, w = m.shape[0] // S, m.shape[1] // S
    a = m.reshape(h, S, w, S).mean((1, 3))
    im = Image.new('RGBA', (w, h), (*rgb, 0))
    im.putalpha(Image.fromarray(np.round(a * 255).astype(np.uint8)))
    return im


# --- pictograms: one Material symbol centered on the stock canvas


PICTOGRAMS = {
    'stat_sys_alarm': 'alarm',
    'stat_sys_alarm_dim': 'alarm',
    'stat_sys_cast': 'cast',
    'stat_sys_data_bluetooth': 'bluetooth',
    'stat_sys_data_bluetooth_connected': 'bluetooth_connected',
    'stat_sys_data_saver': 'data_saver_on',
    'stat_sys_hotspot': 'wifi_tethering',
    'stat_sys_location': 'location_on',
    'stat_sys_nfc': 'nfc',
    'stat_sys_no_sim': 'no_sim',
    'stat_sys_ringer_silent': 'notifications_off',
    'stat_sys_ringer_vibrate': 'mobile_vibrate',
    'stat_sys_vpn_ic': 'vpn_key',
    'stat_sys_signal_flightmode': 'airplanemode_active',
    'ic_statusbar_nodistub': 'bedtime',
    'ic_statusbar_eyecare': 'visibility',
    'ic_performance_turbo': 'speed',
    'ic_stat_statellite': 'satellite_alt',
    'ic_stat_statellite_beidou': 'satellite_alt',
    'ic_statusbar_battery_reverse_charge': 'battery_android_share',
    'ic_statusbar_dial_wifi': 'wifi_calling',
    'stat_sys_incall_vowifi': 'wifi_calling',
    'ic_statusbar_hd_warning': 'phone_disabled',
    'ic_statusbar_phone': 'call',
    'ic_statusbar_synergy': 'devices',
    'stat_sys_airplane_mode': 'airplanemode_active',
    'stat_sys_ethernet_fully': 'settings_ethernet',
    'stat_sys_managed_profile_status': 'work',
    'stat_sys_network_sharing_ic': 'share',
    'stat_sys_rtt': 'rtt',
    'stat_sys_tty_mode': 'tty',
    'stat_sys_zen_important': 'bedtime',
    'stat_sys_zen_none': 'bedtime',
    # MagicOS 11 scene modes (HnIconVectorDrawable, 16.25dp)
    'stat_sys_zen_mode_date': 'calendar_month',
    'stat_sys_zen_mode_default': 'sentiment_satisfied',
    'stat_sys_zen_mode_draw': 'palette',
    'stat_sys_zen_mode_drive': 'directions_car',
    'stat_sys_zen_mode_exam': 'edit_document',
    'stat_sys_zen_mode_family': 'home',
    'stat_sys_zen_mode_flow': 'ecg_heart',
    'stat_sys_zen_mode_game': 'sports_esports',
    'stat_sys_zen_mode_lecture': 'co_present',
    'stat_sys_zen_mode_movie': 'movie',
    'stat_sys_zen_mode_music': 'music_note',
    'stat_sys_zen_mode_personal': 'person',
    'stat_sys_zen_mode_pet': 'pets',
    'stat_sys_zen_mode_read': 'menu_book',
    'stat_sys_zen_mode_sleep': 'bed',
    'stat_sys_zen_mode_social': 'forum',
    'stat_sys_zen_mode_speed_rail': 'train',
    'stat_sys_zen_mode_sport': 'directions_run',
    'stat_sys_zen_mode_theater': 'curtains',
    'stat_sys_zen_mode_work': 'work',
}
EM = 50  # Material's 24-unit box, so a full-height glyph is about 42px


def fit(name: str, w: float, h: float = H) -> Mask:
    """A symbol centered on a w x h canvas, at the pictograms' size relative to the height."""
    return paste(blank(w, h), _symbol(name, EM * h / H), w / 2, h / 2)


# --- mobile signal: bars on the right of an 80px canvas, network type top left, in/out arrows bottom left


type Box = tuple[float, float, float, float]

# The stock bars (MagicOS 11 vectors): (left, top, width), all to the same bottom
BARS4 = [(32.5, 39, 7), (43.25, 33.88, 7), (54.12, 28.62, 7), (65, 18.88, 7)]  # bottom 52, 80px canvas
NULL5 = [(8 + 10 * i, 43 - 9 * i, 6) for i in range(4)] + [(48, 8, 6)]  # bottom 52, no service, 62px canvas


def bars5(w: float) -> list[tuple[float, float, float]]:
    """5 bars ending 8px from the right edge (x 18..78 of 86px, 12..72 of the 80px _small), bottom 50."""
    return [(w - 68 + 13 * i, 41 - 7 * i, 8) for i in range(5)]


def bars(layout: Sequence[tuple[float, float, float]], lit: int, w: float = 80, bottom: float = 52) -> Mask:
    """Bars at the stock positions with Material's fully rounded ends, the first lit ones opaque."""
    m = blank(w)
    return over(
        *[(rrect(m, (x, t, x + bw, bottom), bw / 2), 1.0 if i < lit else DIM) for i, (x, t, bw) in enumerate(layout)]
    )


@cache
def font(size: float) -> ImageFont.FreeTypeFont:
    """Inter Display Bold, size px (at S x)."""
    return ImageFont.truetype(
        DATA / 'inter/InterDisplay-Bold.ttf', round(size * S), layout_engine=ImageFont.Layout.BASIC
    )


CAP = -font(100).getbbox('H', anchor='ls')[1] / (100 * S)  # cap height, in em
TRACK = -0.04  # letter spacing in em: Inter is wider than Honor's condensed status bar font


def text(m: Mask, s: str, x: float, cap_top: float, cap: float, track: float = TRACK) -> tuple[Mask, float]:
    """Set s with its cap height `cap` px and caps starting at cap_top; returns the mask and the right edge."""
    f = font(cap / CAP)
    base = (cap_top + cap) * S
    xs = [x * S]
    for c in s[:-1]:
        xs.append(xs[-1] + f.getlength(c) + track * f.size)

    def run(d: ImageDraw.ImageDraw) -> None:
        for c, cx in zip(s, xs, strict=True):
            d.text((cx, base), c, font=f, fill=255, anchor='ls')

    return draw(m, run), (xs[-1] + f.getlength(s[-1])) / S


def fitted(m: Mask, s: str, x: float, cap_top: float, cap: float) -> Mask:
    """text(), with the cap height shrunk until the ink ends 1px short of the canvas's right edge, as in stock."""
    while True:
        a = text(m, s, x, cap_top, cap)[0]
        if not a[:, -S:].any():
            return a
        cap -= 0.25


def label(main: str, sup: str = '', w: float = 80) -> Mask:
    """A network type like 4G or H+: main caps 19px tall from (4, 8), sup set small and raised after it.

    As in the stock vectors, main shrinks to stay left of the third bar (x=42), and sup, which sits above the bars'
    tops, shrinks on its own to stay left of the fourth (x=53). w is the canvas width (80, or 86 for some).
    """
    m = blank(w)
    cap = 19.0
    while True:
        a, right = text(m, main, 4, 8, cap)
        if right <= 42 or cap < 12:
            break
        cap -= 0.5
    if not sup:
        return a
    small = cap * 0.45
    while True:
        b, end = text(m, sup, right + 0.6, 8, small)
        if end <= 53 or small < 6:
            return np.maximum(a, b)
        small -= 0.25


LABELS = {  # name: network type (main, raised small part); the canvas width comes from SIZES
    'stat_sys_data_connected_5g_cust': ('5G',),
    'stat_sys_data_connected_5g_cust_available': ('5G',),
    'stat_sys_data_connected_c_2g': ('2G',),
    'stat_sys_data_connected_c1_2g': ('2G',),
    'stat_sys_data_connected_c2_2g': ('2G',),
    'stat_sys_data_connected_c_3g': ('3G',),
    'stat_sys_data_connected_g_for_gsm_gprs': ('G',),
    'stat_sys_data_connected_5g_plus_cust': ('5G', '+'),
    'stat_sys_data_fully_connected_5point5g': ('5.5G',),
    'stat_sys_signal_roam_type_2g': ('2G',),
    'stat_sys_data_connected_2g': ('2G',),
    'stat_sys_data_connected_3g': ('3G',),
    'stat_sys_data_connected_4g': ('4G',),
    'stat_sys_data_connected_4glte': ('4G', 'LTE'),
    'stat_sys_data_connected_5g': ('5G',),
    'stat_sys_data_connected_5ga': ('5G', 'A'),
    'stat_sys_data_connected_c_h': ('H',),
    'stat_sys_data_connected_c_hplus': ('H', '+'),
    'stat_sys_data_connected_h': ('H',),
    'stat_sys_data_connected_hplus': ('H', '+'),
    'stat_sys_data_fully_connected_1x': ('2G',),  # stock shows CDMA 1x as 2G
    'stat_sys_data_fully_connected_2g': ('2G',),
    'stat_sys_data_fully_connected_3g': ('3G',),
    'stat_sys_data_fully_connected_3gplus': ('3G', '+'),
    'stat_sys_data_fully_connected_4g': ('4G',),
    'stat_sys_data_fully_connected_4glte_plus': ('4G', 'LTE'),
    'stat_sys_data_fully_connected_4gplus': ('4G', '+'),
    'stat_sys_data_fully_connected_4point5g': ('4.5G',),
    'stat_sys_data_fully_connected_4point5g_plus': ('4.5G', '+'),
    'stat_sys_data_fully_connected_e': ('E',),
    'stat_sys_data_fully_connected_g': ('G',),
    'stat_sys_data_fully_connected_h': ('H',),
    'stat_sys_data_fully_connected_hplus': ('H', '+'),
    'stat_sys_data_fully_connected_lte': ('LTE',),
    'stat_sys_data_fully_connected_lte_plus': ('LTE', '+'),
    'stat_sys_data_fully_connected_lteplus': ('LTE', '+'),
    'stat_sys_signal_card1_roam_type_2g': ('2G',),
    'stat_sys_signal_card1_roam_type_3g': ('3G',),
    'stat_sys_signal_card1_roam_type_4g': ('4G',),
    'stat_sys_signal_roam_type_3g': ('3G',),
    'stat_sys_signal_roam_type_4g': ('4G',),
    'stat_sys_signal_roam_type_h': ('H',),
    'stat_sys_signal_roam_type_hplus': ('H', '+'),
    'stat_sys_signal_type_2g': ('2G',),
}


def roam(w: float) -> Mask:
    return text(blank(w), 'R', 43, 6, 10)[0]


def arrows(w: float, x: float, up: float, down: float) -> Mask:
    """Data activity: Material's arrow_drop_up and arrow_drop_down in the stock triangles' 9.75px boxes at x."""
    m = blank(w)
    tri_up = paste(m, symbol('arrow_drop_up', w=9.75), x + 4.875, 39.625)
    tri_down = paste(m, symbol('arrow_drop_down', w=9.75), x + 4.875, 47.375)
    return over((tri_up, up), (tri_down, down))


def bang(m: Mask, box: Box, dot: float) -> Mask:
    """An exclamation mark: a pill over box and a round dot below it, dot px across, as Material draws them."""
    x0, y0, x1, y1 = box
    r = (x1 - x0) / 2
    cx = x0 + r
    d = draw(m, lambda dr: dr.ellipse(((cx - dot / 2) * S, y1 * S, (cx + dot / 2) * S, (y1 + dot) * S), fill=255))
    return np.maximum(d, rrect(m, (x0, y0, x1, y1 - 2), r))


# --- wifi: Material's android_wifi_4_bar (a dot and three arcs) in the stock glyph's box, lit from the dot up


WIFI_BOX = (7, 9, 65.62, 50.62)  # the stock glyph, the same in the 71 and 73px canvases


@cache
def arcs(w: float) -> list[Mask]:
    """The glyph's parts, dot first; each antialiased edge pixel goes to the nearest part."""
    x0, y0, x1, y1 = WIFI_BOX
    m = paste(blank(w), symbol('android_wifi_4_bar', h=y1 - y0), (x0 + x1) / 2, (y0 + y1) / 2)
    lab, n = ndimage.label(m > 0.5)
    near = ndimage.distance_transform_edt(lab == 0, return_indices=True)[1]
    lab = lab[near[0], near[1]]
    parts = [m * (lab == i) for i in range(1, n + 1)]
    return sorted(parts, key=lambda p: -np.nonzero(p.any(1))[0][0])


def wifi(level: int, w: float) -> Mask:
    return over(*[(p, 1.0 if i < level else DIM) for i, p in enumerate(arcs(w))])


def no_internet() -> Mask:
    return bang(blank(71), (58, 30.5, 63, 45.5), 5)  # the stock mark: 58..63, bar to 43.5, dot 45.5..50.5


# --- battery: an outline with a level bar inside. The outline is as big as the stock battery (y 8..52), because
# SystemUI can draw the percentage inside the battery, in a 27.7dp box laid over the whole icon


BATT_W = 104
# The stock layout: body, a separate nub, and the level bar 8px inside the body. Drawn with Material's
# battery_android_frame_*: a 2U outline (U px per 1/24 of the pictograms' box), corners a quarter of the height,
# the level concentric with them, the nub flat on the body side and rounded outside
BODY = (10.0, 8.0, 89.0, 52.0)  # outer edge of the outline
NUB = (92.0, 22.0, 98.0, 38.0)
U = EM / 24
STROKE = 2 * U
INSET = 8 - STROKE  # gap between the outline and the level bar
RADIUS = (BODY[3] - BODY[1]) / 4
BODY_CX = (BODY[0] + BODY[2]) / 2


def frame() -> Mask:
    m = blank(BATT_W)
    x0, y0, x1, y1 = BODY
    outer = rrect(m, BODY, RADIUS)
    inner = rrect(m, (x0 + STROKE, y0 + STROKE, x1 - STROKE, y1 - STROKE), RADIUS - STROKE)
    nub = draw(
        m,
        lambda d: d.rounded_rectangle(
            tuple(v * S for v in NUB), (NUB[2] - NUB[0]) * 2 / 3 * S, fill=255, corners=(False, True, True, False)
        ),
    )
    return np.maximum(outer * (1 - inner), nub)


def level(pct: int) -> Mask:
    e = STROKE + INSET
    x0, y0, x1, y1 = BODY[0] + e, BODY[1] + e, BODY[2] - e, BODY[3] - e
    w = (x1 - x0) * pct / 100
    return rrect(blank(BATT_W), (x0, y0, x0 + w, y1), min(RADIUS - e, w / 2))


# --- charge bolts: Material has no zigzag bolt like Honor's fast and super fast ones, so every bolt is Honor's own
# shape (data/statusbar/honor, extracted from the MagicOS 11 vectors) with its corners rounded as Material Symbols
# Rounded rounds them


ROUND = 1.2  # corner radius, px: Material's rounded corners at the pictograms' scale
BOLTS = [
    f'ic_statusbar_battery_{n}'
    for n in (
        'charge',
        'charge_double',
        'supercharge',
        'charge_figure',
        'charge_double_figure',
        'supercharge_figure',
        *(f'charge{k}{s}_v1' for k in ('', '_fast', '_super_fast') for s in ('', '_single', '_hollow')),
        *(f'charge{k}_hollow_transparent_back_v1' for k in ('', '_fast', '_super_fast')),
    )
]


def rounded(m: Mask, r: float = ROUND) -> Mask:
    """m with its convex corners rounded to r px (an opening) and its concave ones to r/2 (a closing)."""

    def grow(a: Mask | npt.NDArray[np.bool_], by: float) -> Mask:
        return np.clip(by * S + 0.5 - ndimage.distance_transform_edt(a < 0.5), 0, 1)

    opened = grow(grow(1 - m, r) < 0.5, r)
    return 1 - grow(grow(opened, r / 2) < 0.5, r / 2)


@cache
def honor(name: str) -> Mask:
    """A stock shape from data/statusbar/honor, rendered at its viewport size and rounded."""
    svg = DATA / f'statusbar/honor/{name}.svg'
    w, h = SIZES[name]
    png_ = resvg_py.svg_to_bytes(svg_path=str(svg), width=round(w * S), height=round(h * S))
    return rounded(np.asarray(Image.open(io.BytesIO(bytes(png_))).getchannel('A')) / 255)


# --- battery, the new style (_v1). SystemUI uses it unless a theme replaces only the old drawables; Honor's
# HwBatteryManager.checkThemeBatteryIconStyle. Two looks: a solid battery with the level cut from it and the
# percentage drawn inside by SystemUI (*_full_v1, charge_*_v1 next to the number), or an outline with a level bar
# (border_hollow_v1, *_v1, charge_*_hollow_v1). The stock layout, drawn like the old style above


BODY_V1 = (10.0, 8.0, 90.5, 52.0)
NUB_V1 = (93.5, 23.0, 98.0, 37.0)
LEVEL_V1 = (16.0, 14.0, 84.5, 46.0)  # the hollow style's level bar at 100%
RADIUS_V1 = (BODY_V1[3] - BODY_V1[1]) / 4


def nub_v1(m: Mask) -> Mask:
    r = (NUB_V1[2] - NUB_V1[0]) * 2 / 3
    return draw(
        m,
        lambda d: d.rounded_rectangle(
            tuple(v * S for v in NUB_V1), r * S, fill=255, corners=(False, True, True, False)
        ),
    )


def solid_v1(pct: int = 100) -> Mask:
    """The solid battery, cut at pct of the body width (the nub only at 100)."""
    m = blank(BATT_W)
    body = rrect(m, BODY_V1, RADIUS_V1)
    x = BODY_V1[0] + (BODY_V1[2] - BODY_V1[0]) * pct / 100
    body[:, round(x * S) :] = 0
    return np.maximum(body, nub_v1(m)) if pct == 100 else body


def hollow_v1() -> Mask:
    m = blank(BATT_W)
    x0, y0, x1, y1 = BODY_V1
    outer = rrect(m, BODY_V1, RADIUS_V1)
    inner = rrect(m, (x0 + STROKE, y0 + STROKE, x1 - STROKE, y1 - STROKE), RADIUS_V1 - STROKE)
    return np.maximum(outer * (1 - inner), nub_v1(m))


def level_v1(pct: int) -> Mask:
    x0, y0, x1, y1 = LEVEL_V1
    w = (x1 - x0) * pct / 100
    return rrect(blank(BATT_W), (x0, y0, x0 + w, y1), min(RADIUS_V1 - (x0 - BODY_V1[0]), w / 2))


def battery_v1() -> dict[str, Mask]:
    out = {'ic_statusbar_battery_v1': solid_v1(), 'ic_statusbar_battery_border_hollow_v1': hollow_v1()}
    for pct in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100):
        out[f'ic_statusbar_battery_{pct}_full_v1'] = solid_v1(pct)
        out[f'ic_statusbar_battery_{pct}_v1'] = level_v1(pct)
    return out


# --- bluetooth device battery: a device glyph left of the stock battery outline (the stock level drawables stay)


def device(w: float, glyph: str, check: bool = False) -> Mask:
    """As in the stock vectors: a battery at x = w-30..w-8 with a separate nub, and the glyph in a 42px square
    from x=10 (y 9..51). The battery has Material's rounded corners."""
    m = blank(w)
    x = w - 30
    body = rrect(m, (x, 14.25, x + 22, 51.25), 22 / 4)
    nub = draw(
        m,
        lambda d: d.rounded_rectangle(
            ((x + 7) * S, 9.12 * S, (x + 15) * S, 12.25 * S), U * S, fill=255, corners=(True, True, False, False)
        ),
    )
    g = _symbol(glyph, 42 * 24 / 20)  # Material's 20-unit live area fills the 42px square
    gx = (10 + x - 6) / 2
    out = over((np.maximum(body, nub), 0.5), (paste(m, g, gx, 30), 1.0))
    if check:
        out = badge(out, gx + g.shape[1] / S / 2 - 5, 45)
    return out


def badge(m: Mask, cx: float, cy: float) -> Mask:
    """A check badge with a clear ring around it."""
    ring = draw(m, lambda d: d.ellipse(((cx - 10) * S, (cy - 10) * S, (cx + 10) * S, (cy + 10) * S), fill=255))
    return np.maximum(m * (1 - ring), paste(m, symbol('check_circle', h=16), cx, cy))


def standalone(glyph: str, w: float, check: bool = False) -> Mask:
    g = _symbol(glyph, EM)
    out = paste(blank(w), g, w / 2, H / 2)
    if check:
        out = badge(out, w / 2 + g.shape[1] / S / 2 - 7, 46)
    return out


# --- calling: VoLTE and the like drawn as One UI does, "Vo" with two waves over the rest of the word and the SIM number
# right after it ("Vo)) / LTE1"); HD and WLAN keep a filled tag with the text knocked out
#
# SystemUI puts the SIM number in its own drawable right of the tag, and the same number drawables follow VoLTE and HD.
# With one SIM it shows *_superscript_icon1 or 2, with both on the same kind *_superscript_icon_double (1 over 2).

CALL_CAP = 17  # cap height of each line, the SIM numbers' too
CALL_GAP = 0.32  # between the two lines, in caps
CALL_TRACK = -0.06  # One UI sets these tighter than the network types
WAVE = 0.17  # stroke of the waves, in caps: about the letters' stems
RING = 0.18  # from the o to the first wave and between the waves, in caps
EDGE = 0.5  # the tag's ink ends this far from its right edge


def call_rows(h: float) -> tuple[float, float, float]:
    """(cap, top of line 1, top of line 2) for a 60px canvas, scaled for the 36px ones."""
    cap = CALL_CAP * h / H
    top1 = (h - cap * (2 + CALL_GAP)) / 2
    return cap, top1, top1 + cap * (1 + CALL_GAP)


def ink(a: Mask, axis: int = 0) -> tuple[float, float]:
    """The first and last px with ink, columns by default, rows with axis=1."""
    idx = np.nonzero(a.max(axis) > 0.5)[0]
    return idx.min() / S, (idx.max() + 1) / S


def shifted(a: Mask, dx: float) -> Mask:
    return np.roll(a, round(dx * S), axis=1)


def waves(m: Mask, cx: float, cy: float, radii: Sequence[float], sw: float, span: float) -> Mask:
    """Arcs of these radii around (cx, cy), sw px thick with round ends, opening right over +-span degrees."""

    def run(d: ImageDraw.ImageDraw) -> None:
        for r in radii:
            o = (r + sw / 2) * S
            d.arc((cx * S - o, cy * S - o, cx * S + o, cy * S + o), -span, span, fill=255, width=round(sw * S))
            for t in np.radians((-span, span)):
                x, y = cx + r * np.cos(t), cy + r * np.sin(t)
                d.ellipse(((x - sw / 2) * S, (y - sw / 2) * S, (x + sw / 2) * S, (y + sw / 2) * S), fill=255)

    return draw(m, run)


def vo(w: float, h: float, rest: str) -> Mask:
    """Vo with two waves around the o over `rest`. The second line ends at the right edge so the SIM number follows
    it like a letter; the first is centered over it but stops at that edge too."""
    m = blank(w, h)
    cap, top1, top2 = call_rows(h)
    a = text(m, 'Vo', 0, top1, cap, CALL_TRACK)[0]
    left = ink(a)[0]
    f = font(cap / CAP)
    o = text(m, 'o', (f.getlength('V') + CALL_TRACK * f.size) / S, top1, cap, CALL_TRACK)[0]
    (ol, or_), (ot, ob) = ink(o), ink(o, 1)
    cx, cy = (ol + or_) / 2, (ot + ob) / 2
    sw = WAVE * cap
    r1 = (or_ - ol) / 2 + RING * cap + sw / 2
    r2 = r1 + sw + RING * cap
    # as tall as One UI's, unless the outer wave would come within 1.5px of the second line
    span = min(40, np.degrees(np.arcsin(min(1, (top2 - 1.5 - sw / 2 - cy) / r2))))
    a = np.maximum(a, waves(m, cx, cy, (r1, r2), sw, span))
    b = text(m, rest, 0, top2, cap, CALL_TRACK)[0]
    bl, br = ink(b)
    b = shifted(b, w - EDGE - br)
    right = ink(a)[1]
    dx = min((bl + br) / 2 + w - EDGE - br - (left + right) / 2, w - EDGE - right)
    return np.maximum(shifted(a, dx), b)


VO = {  # name: what goes under Vo
    'ic_statusbar_volte': 'LTE',
    'ic_statusbar_vo4g': '4G',
    'ic_statusbar_vo5g': '5G',
    'ic_statusbar_vonr': 'NR',
    'ic_statusbar_vowifi': 'WiFi',
    'ic_statusbar_voz4g': 'z4G',
    'ic_statusbar_incall_double_volte': 'LTE',
    'ic_statusbar_incall_double_vo5g': '5G',
    'ic_statusbar_incall_double_vonr': 'NR',
    'ic_statusbar_incall_double_vowifi': 'WiFi',
}


def sim_numbers(h: float, rows: Sequence[tuple[str, int]]) -> Mask:
    """SIM numbers in their 19px canvas, ink from x=0.8: (digit, line 0 or 1) for each, set like the tag's lines."""
    m = blank(19, h)
    cap, *tops = call_rows(h)
    out = np.zeros_like(m)
    for d, row in rows:
        a = text(m, d, 0, tops[row], cap, CALL_TRACK)[0]
        out = np.maximum(out, shifted(a, 0.8 - ink(a)[0]))
    return out


def tag(w: float, h: float, top: float, bottom: float, main: str, sup: str = '') -> Mask:
    """A rounded tag over the full width from top to bottom, with main (and a small raised sup) cut out of it."""
    m = blank(w, h)
    box = rrect(m, (0, top, w, bottom), 2 * U)  # the corners of Material's *_mobiledata_badge
    cap = (bottom - top) * 0.62
    while True:
        a, right = text(m, main, 0, 0, cap)
        width = right
        if sup:
            b, width = text(m, sup, right + 0.4, 0, cap * 0.55)
            a = np.maximum(a, b)
        if width <= w - (bottom - top) * 0.22 or cap < 6:
            break
        cap -= 0.25
    # center the text block: shift the mask
    dx = round(((w - width) / 2) * S)
    dy = round((top + (bottom - top - cap) / 2) * S)
    cut = np.zeros_like(m)
    cut[dy:, dx:] = a[: m.shape[0] - dy, : m.shape[1] - dx]
    return box * (1 - cut)


TAGS = {  # name: (canvas w, h, tag top, bottom, text)
    'ic_statusbar_hd': (44, 60, 10, 50, ('HD',)),
    'ic_statusbar_hd_plus': (54, 60, 8, 52, ('HD', '+')),
    'ic_statusbar_hd_plus_cmcc': (44, 60, 8, 52, ('HD', '+')),
    'ic_statusbar_incall_wlan': (66, 60, 8, 52, ('WLAN',)),
    'stat_sys_hdlt': (60, 60, 8, 52, ('HD',)),
    'stat_sys_hd_plus': (65, 60, 8, 52, ('HD', '+')),
    'ic_statusbar_incall_double_hd': (68, 36, 5, 31, ('HD',)),
    'ic_statusbar_incall_double_hd_plus_cmcc': (68, 36, 5, 31, ('HD', '+')),
    'ic_statusbar_incall_double_wlan': (59, 36, 4, 32, ('WLAN',)),
}


def ethernet_alert() -> Mask:
    """Ethernet without internet: the symbol moved left, an exclamation mark after it."""
    m = blank(53, 48)
    g = paste(m, symbol('settings_ethernet', w=44.5), 24, 24)  # the stock glyph spans x 1.75..46.25, y 11..37
    return np.maximum(g, bang(m, (49, 20, 53, 38), 4))  # the stock mark: 49..53, bar to 36, dot 40..44


# --- everything


def icons() -> dict[str, Image.Image]:
    out: dict[str, Image.Image] = {}

    def width(name: str) -> float:
        return SIZES[name][0]

    for name, glyph in PICTOGRAMS.items():
        out[name] = png(fit(glyph, *SIZES[name]))
    out['stat_sys_ethernet'] = png(ethernet_alert())

    for lit in range(5):
        for suffix in ('', '_fully'):
            out[f'stat_sys_signal_{lit}{suffix}'] = png(bars(BARS4, lit))
    for lit in range(6):
        for prefix in ('stat_sys_signal_five_', 'ic_statusbar_signal_tjt_'):
            for suffix in ('', '_fully', '_small'):
                name = f'{prefix}{lit}{suffix}'
                out[name] = png(bars(bars5(width(name)), lit, w=width(name), bottom=50))
    for name, parts in LABELS.items():
        out[name] = png(label(*parts, w=width(name)))
    # a tiny standalone LTE+ (24 x 24 in stock)
    m = blank(24, 24)
    out['stat_sys_data_fully_connected_lte_plus'] = png(fitted(m, 'LTE+', 1, 3, 8))
    for name, (w, h, top, bottom, parts) in TAGS.items():
        out[name] = png(tag(w, h, top, bottom, *parts))
    for name, rest in VO.items():
        out[name] = png(vo(*SIZES[name], rest))
    # a lone number sits on the second line (LTE1), both stack on the two lines
    for h, prefix in ((36, 'ic_statusbar_superscript_double_icon'), (60, 'ic_statusbar_superscript_icon')):
        out[f'{prefix}1'] = png(sim_numbers(h, [('1', 1)]))
        out[f'{prefix}2'] = png(sim_numbers(h, [('2', 1)]))
    out['ic_statusbar_superscript_icon_double'] = png(sim_numbers(60, [('1', 0), ('2', 1)]))
    out['ic_statusbar_wifi_giga'] = png(fitted(blank(12, 63), 'G', 0, 8, 18))
    m = blank(62)
    out['stat_sys_signal_five_null'] = png(np.maximum(bars(NULL5, 0, w=62), paste(m, symbol('close', w=16), 16, 16)))
    m = blank(26, 24)
    out['stat_sys_wifi_signal_null'] = png(paste(m, symbol('signal_wifi_0_bar', w=25.12), 12.94, 11.94))
    for name in ('stat_sys_data_connected_roam', 'stat_sys_signal_type_roam'):
        out[name] = png(roam(width(name)))
    out['stat_sys_signal_in'] = png(arrows(80, 6.62, DIM, 1))
    out['stat_sys_signal_out'] = png(arrows(80, 6.62, 1, DIM))
    out['stat_sys_signal_inout'] = png(arrows(80, 6.62, 1, 1))
    out['single_stat_sys_signal_connected'] = png(arrows(80, 6.62, DIM, DIM))
    m = blank(67)
    out['stat_sys_signal_type_waiting'] = png(paste(m, symbol('hourglass_empty', h=18), 15, 17))
    m = blank(40, 40)
    out['stat_sys_data_disabled'] = png(paste(m, symbol('close', h=13), 12, 12))

    kinds = ('wifi', 'wifi2', 'wifi2_6', 'wifi2_6plus', 'wifi2_7', 'wifi6', 'wifi6_plus', 'wifi7', 'wifi7_plus')
    for lvl in range(5):
        names = [f'stat_sys_{k}_signal_{lvl}' for k in kinds]
        names += [f'stat_sys_wifi_signal_{lvl}_fully', f'stat_sys_charged_wifi_signal_{lvl}']
        for name in names:
            out[name] = png(wifi(lvl, width(name)))
    out['stat_sys_wifi_in'] = png(arrows(71, 52.62, DIM, 1))
    out['stat_sys_wifi_out'] = png(arrows(71, 52.62, 1, DIM))
    out['stat_sys_wifi_inout'] = png(arrows(71, 52.62, 1, 1))
    out['stat_sys_wifi_none'] = png(arrows(71, 52.62, DIM, DIM))
    out['stat_sys_wifi_no_internet'] = png(no_internet())
    m = blank(73)
    out['stat_sys_wifi_hotspot'] = png(paste(m, symbol('wifi_tethering', h=16), 11, 45))

    out['ic_statusbar_battery'] = png(frame())
    for pct in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100):
        # SystemUI leaves levels <= 20% untinted, so they show their own color
        rgb = CRITICAL if pct <= 10 else LOW if pct <= 20 else (0, 0, 0)
        im = png(level(pct), rgb)
        for suffix in ('', '_charge', '_full', '_save'):
            out[f'ic_statusbar_battery_{pct}{suffix}'] = im
    out.update({name: png(honor(name)) for name in BOLTS})
    out['ic_statusbar_battery_powersaving'] = png(paste(blank(BATT_W), symbol('add', h=20), BODY_CX, 30))
    out['ic_statusbar_battery_powersaving_figurein'] = png(paste(blank(36), symbol('add', h=18), 18, 30))
    out.update({name: png(m) for name, m in battery_v1().items()})

    for name, glyph, check in (
        ('ic_hnstatusbar_earphone_battery', 'headphones', False),
        ('ic_hnstatusbar_earservices_battery', 'hearing', False),
        ('ic_hnstatusbar_loudspeaker_battery', 'speaker', False),
        ('ic_hnstatusbar_bluetooth_device_default', 'bluetooth', False),
        ('stat_sys_data_bluetooth_register_battery_bg', 'headphones', True),
        ('stat_sys_data_bluetooth_unregister_battery_bg', 'headphones', False),
    ):
        out[name] = png(device(width(name), glyph, check))
    for name, check in (
        ('stat_sys_data_bluetooth_voice_id_connected', False),
        ('stat_sys_data_bluetooth_voice_id_registered', True),
    ):
        out[name] = png(standalone('headphones', width(name), check))

    # SystemUI stretches each PNG to the stock drawable's bounds, so the sizes must match
    wrong = {n: im.size for n, im in out.items() if im.size != tuple(round(v) for v in SIZES[n])}
    if wrong or out.keys() != SIZES.keys():
        raise SystemExit(
            f'size mismatch {wrong}, missing {SIZES.keys() - out.keys()}, extra {out.keys() - SIZES.keys()}'
        )
    return out


def files() -> dict[str, bytes]:
    """The two module zips, the same for both themes; also writes the PNGs to build/statusbar for review."""
    res = BUILD / 'statusbar'
    res.mkdir(parents=True, exist_ok=True)
    ui, status = MODULES
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(buf := io.BytesIO(), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(DATA / 'statusbar/theme.xml', 'theme.xml')
        # SystemUI only checks for this exact entry; night/theme.xml is empty, so dark mode uses the root values too
        zf.writestr('night/', '')
        zf.writestr('night/theme.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<resources />\n')
    out[ui] = buf.getvalue()
    with zipfile.ZipFile(buf := io.BytesIO(), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('res/', '')
        zf.writestr('res/drawable-xxhdpi/', '')
        for name, im in sorted(icons().items()):
            png = io.BytesIO()
            im.save(png, 'PNG')
            (res / f'{name}.png').write_bytes(png.getvalue())
            zf.writestr(f'res/drawable-xxhdpi/{name}.png', png.getvalue())
    out[status] = buf.getvalue()
    return out


def build(dst: Path, files: dict[str, bytes]) -> None:
    """Write the two module zips into the directory dst."""
    for name, data in files.items():
        (dst / name).write_bytes(data)


if __name__ == '__main__':
    build(BUILD, files())
    print(len(list((BUILD / 'statusbar').glob('*.png'))), 'icons')
