# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['oi_viewer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icon.png', '.'),
    ],
    hiddenimports=[
        'pandas_market_calendars',
        'exchange_calendars',
        'tkcalendar',
        'babel.numbers',
        'babel.dates',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'pytz',
        'charset_normalizer',
        'certifi',
        'multitasking',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='OptionsView',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='OptionsView.app',
    icon=None,
    bundle_identifier='com.moopertonic.optionsview',
)
