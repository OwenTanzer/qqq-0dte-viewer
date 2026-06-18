#!/usr/bin/env python3
"""
Build OptionsView.exe (Windows) or OptionsView (Mac/Linux).

    python build.py

Requires: pyinstaller, Pillow (pip install pyinstaller Pillow)
"""
import subprocess
import sys
from pathlib import Path

ico = Path("icon.ico")
png = Path("icon.png")

if not ico.exists():
    try:
        from PIL import Image
        img = Image.open(png)
        img.save(ico, format="ICO",
                 sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        print(f"Created {ico}")
    except ImportError:
        print("Pillow not found — install it with:  pip install Pillow")
        sys.exit(1)

subprocess.run(
    [sys.executable, "-m", "PyInstaller", "--clean", "OptionsView.spec"],
    check=True,
)
print("\nDone — binary is in dist/OptionsView.exe")
