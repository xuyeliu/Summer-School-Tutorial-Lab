"""Generate the QR codes that point participants at the lab.

Run this after changing LAB_URL:

    pip install segno
    python make_qr.py

Outputs land in docs/qr/, so they are also downloadable over GitHub Pages
(https://xuyeliu.github.io/Summer-School-Tutorial-Lab/qr/qr_huge.png) and you do not
have to carry image files to the machine you present from.
"""

import os

import segno

# The GitHub Pages landing page, not the Colab URL directly. Two reasons: it is short
# enough to read aloud and type, and it survives the notebook being renamed or moved.
LAB_URL = 'https://xuyeliu.github.io/Summer-School-Tutorial-Lab/'
SPOKEN_URL = 'xuyeliu.github.io/Summer-School-Tutorial-Lab'

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'qr')

# Error correction M (15%) is the sweet spot here. Higher levels add modules, which makes
# every module physically smaller at a fixed print size, and a projected QR fails from
# small modules long before it fails from missing error correction.
ERROR_LEVEL = 'm'

# The quiet zone is not decoration. Most scanners refuse a code without 4 clear modules
# around it, which is exactly what breaks when a QR is pasted flush into a slide corner.
BORDER = 4

TARGETS = [
    ('qr_huge', 2400),    # opening slide, full screen
    ('qr_corner', 800),   # top-right corner of every slide after that
]


def scale_for(qr, target_px):
    """Integer module scale that lands closest to the requested pixel width."""
    modules = qr.symbol_size(border=BORDER)[0]
    return max(1, round(target_px / modules))


def write_labeled(qr, path, scale):
    """The opening slide needs the URL spelled out under the code, for anyone not scanning."""
    from PIL import Image, ImageDraw, ImageFont

    qr_img = Image.new('RGB', (1, 1))
    tmp = path + '.tmp.png'
    qr.save(tmp, scale=scale, border=BORDER, dark='black', light='white')
    qr_img = Image.open(tmp).convert('RGB')
    os.remove(tmp)

    width = qr_img.width
    font_px = max(16, width // 26)
    font = _load_font(font_px)

    pad_top = font_px // 2
    pad_bottom = font_px
    canvas = Image.new('RGB', (width, qr_img.height + pad_top + font_px + pad_bottom), 'white')
    canvas.paste(qr_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), SPOKEN_URL, font=font)
    draw.text(((width - (right - left)) // 2 - left, qr_img.height + pad_top - top),
              SPOKEN_URL, fill='black', font=font)
    canvas.save(path)
    return canvas.size


def _load_font(size_px):
    from PIL import ImageFont

    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ]
    try:
        import matplotlib
        candidates.append(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf',
                                       'DejaVuSans-Bold.ttf'))
    except ImportError:
        pass
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size_px)
            except OSError:
                continue
    print('  WARNING: no scalable font found, the URL caption will be tiny')
    return ImageFont.load_default()


def verify(paths):
    """Decode what we just wrote and confirm it round-trips to LAB_URL.

    Also decodes the corner code shrunk to 100px, which is roughly what a 1 inch element
    on a projected slide gives a phone camera.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        print('\ncv2 not installed, skipping the decode check')
        return

    detector = cv2.QRCodeDetector()
    print()
    for path in paths:
        decoded, _, _ = detector.detectAndDecode(cv2.imread(path))
        status = 'OK ' if decoded == LAB_URL else 'FAIL'
        print(f'  [{status}] decoded {os.path.basename(path)}')
        assert decoded == LAB_URL, f'{path} decoded to {decoded!r}, expected {LAB_URL!r}'

    small = Image.open(os.path.join(OUT_DIR, 'qr_corner.png')).resize((100, 100),
                                                                     Image.LANCZOS)
    decoded, _, _ = detector.detectAndDecode(np.array(small.convert('RGB'))[:, :, ::-1])
    assert decoded == LAB_URL, f'corner code failed at 100px, decoded {decoded!r}'
    print('  [OK ] decoded qr_corner.png shrunk to 100px (projected-slide size)')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    qr = segno.make(LAB_URL, error=ERROR_LEVEL)
    print(f'{LAB_URL}\n  QR version {qr.version}, '
          f'{qr.symbol_size(border=BORDER)[0]} modules per side including the quiet zone\n')

    png_paths = []
    for name, target_px in TARGETS:
        scale = scale_for(qr, target_px)
        png = os.path.join(OUT_DIR, f'{name}.png')
        svg = os.path.join(OUT_DIR, f'{name}.svg')
        # light= keeps the background opaque white. A transparent QR is unscannable on
        # any dark-background slide, which is the classic way this goes wrong in a talk.
        qr.save(png, scale=scale, border=BORDER, dark='black', light='white')
        qr.save(svg, scale=scale, border=BORDER, dark='black', light='white')
        px = qr.symbol_size(scale=scale, border=BORDER)[0]
        print(f'  {name}.png / .svg  {px}x{px} px (scale {scale})')
        png_paths.append(png)

    labeled = os.path.join(OUT_DIR, 'qr_huge_labeled.png')
    size = write_labeled(qr, labeled, scale_for(qr, dict(TARGETS)['qr_huge']))
    print(f'  qr_huge_labeled.png  {size[0]}x{size[1]} px (URL caption underneath)')

    verify(png_paths)


if __name__ == '__main__':
    main()
