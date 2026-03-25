# -*- mode: python ; coding: utf-8 -*-
import os
import cv2
import mediapipe

haarcascades_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

# mediapipe の tasks/c ディレクトリ（ネイティブライブラリを含む）
mediapipe_dir = os.path.dirname(mediapipe.__file__)
mediapipe_tasks_c_dir = os.path.join(mediapipe_dir, 'tasks', 'c')

a = Analysis(
    ['face_blur_stream.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('blaze_face_short_range.tflite', '.'),
        ('efficientdet_lite0.tflite', '.'),
        (os.path.join(haarcascades_dir, 'haarcascade_russian_plate_number.xml'),
         os.path.join('cv2', 'data')),
        (mediapipe_tasks_c_dir, os.path.join('mediapipe', 'tasks', 'c')),
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.c',
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
