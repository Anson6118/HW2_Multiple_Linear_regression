"""Chromium DOM smoke tests plus desktop/mobile screenshots of the real page."""
from pathlib import Path
import subprocess
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_pdf import ROOT,find_browser


def main():
    source=(ROOT/'web/dist/index.html').read_text(encoding='utf-8')
    source=source.replace('<head>','<head><base href="../web/dist/">',1)
    source=source.replace('</head>','<script src="../../tests/browser_smoke.js" defer></script></head>')
    smoke=ROOT/'tests/browser_smoke.html'
    smoke.write_text(source,encoding='utf-8')
    node=shutil.which('node')
    if not node:raise RuntimeError('Node is required for browser integration tests.')
    subprocess.run([node,str(ROOT/'tests/browser_cdp.cjs'),find_browser()],cwd=ROOT,check=True,timeout=45)


if __name__=='__main__':main()
