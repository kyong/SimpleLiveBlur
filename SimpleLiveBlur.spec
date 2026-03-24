# -*- mode: python ; coding: utf-8 -*-
import os
import cv2

# OpenCV の Haar Cascade パスを取得
haarcascades_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

a = Analysis(
    ['face_blur_stream.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('blaze_face_short_range.tflite', '.'),
        ('efficientdet_lite0.tflite', '.'),
        (os.path.join(haarcascades_dir, 'haarcascade_russian_plate_number.xml'),
         os.path.join('cv2', 'data')),
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SimpleLiveBlur',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SimpleLiveBlur',
)

# macOS .app バンドル
app = BUNDLE(
    coll,
    name='SimpleLiveBlur.app',
    icon=None,
    bundle_identifier='com.simplyliveblur.app',
    info_plist={
        'NSCameraUsageDescription': 'SimpleLiveBlur needs camera access for real-time face blurring.',
        'CFBundleShortVersionString': '1.0.0',
    },
)
