"""Download and verify the FOUR distinct, openly licensed HAWIYA gallery photos.

No data is uploaded. Sources are public-domain / CC0 and recorded in IMAGE_SOURCES.txt.
The downloaded photos become local assets; the gallery does not depend on hotlinked URLs.
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import shutil
import urllib.request
from PIL import Image

SOURCES = {
    'green': ('https://upload.wikimedia.org/wikipedia/commons/4/42/GreenEye3.jpg',
              'https://commons.wikimedia.org/wiki/File:GreenEye3.jpg'),
    'brown': ('https://upload.wikimedia.org/wikipedia/commons/e/e5/Brown_eye.jpg',
              'https://commons.wikimedia.org/wiki/File:Brown_eye.jpg'),
    'amber': ('https://upload.wikimedia.org/wikipedia/commons/3/3e/A_human_male_brown_eye.jpg',
              'https://commons.wikimedia.org/wiki/File:A_human_male_brown_eye.jpg'),
    'hazel': ('https://upload.wikimedia.org/wikipedia/commons/8/8d/Eye_closeup.jpg',
              'https://commons.wikimedia.org/wiki/File:Eye_closeup.jpg'),
}


def valid_jpeg(data: bytes) -> None:
    if len(data) < 10000 or len(data) > 10 * 1024 * 1024:
        raise ValueError(f'Unexpected JPEG length: {len(data)} bytes')
    with Image.open(io.BytesIO(data)) as test:
        if test.format != 'JPEG' or min(test.size) < 400:
            raise ValueError(f'Not a suitable JPEG: {test.format}, {test.size}')
        test.verify()


def ensure_gallery(root: Path) -> Path:
    root = Path(root).resolve()
    folder = root / 'static' / 'gallery'
    folder.mkdir(parents=True, exist_ok=True)
    # If manual downloads are needed, place the four named JPEGs in this folder.
    for key, (url, info) in SOURCES.items():
        dst = folder / f'{key}.jpg'
        if dst.is_file():
            try:
                valid_jpeg(dst.read_bytes())
                print(f'PHOTO READY: {key} ({dst.stat().st_size} bytes)', flush=True)
                continue
            except Exception:
                print(f'INVALID existing photo {dst}; downloading a replacement.', flush=True)
        print(f'DOWNLOADING {key.upper()}: {url}', flush=True)
        try:
            req=urllib.request.Request(url, headers={
                'User-Agent':'Mozilla/5.0 HAWIYA/1.0 educational project',
                'Accept':'image/jpeg,image/*;q=0.9,*/*;q=0.8',
            })
            with urllib.request.urlopen(req, timeout=35) as response:
                raw = response.read(10 * 1024 * 1024 + 1)
            valid_jpeg(raw)
            tmp = dst.with_name(dst.name + '.download')
            tmp.write_bytes(raw)
            tmp.replace(dst)
            print(f'PHOTO SAVED: {key} ({len(raw)} bytes)', flush=True)
        except Exception as exc:
            raise RuntimeError(
                f'Could not download {key}: {exc}\n'
                f'Open this public photo in Chrome:\n  {url}\n'
                f'Save it as {dst.name} inside:\n  {folder}\n'
                f'Official source and license: {info}\n'
                'Then run the same command again.'
            ) from exc
    return folder


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    ensure_gallery(args.project)
    print('SUCCESS: Four DIFFERENT real iris photos are stored locally.',flush=True)
