"""Print local reports to PDFs using an installed Chromium; no network needed."""
from pathlib import Path
import argparse
import os
import subprocess
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def find_browser():
    paths=[r'C:\Program Files\Google\Chrome\Application\chrome.exe',
           r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
           r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe']
    for p in paths:
        if Path(p).is_file():return p
    for name in ['chromium','chromium-browser','google-chrome','msedge']:
        p=shutil.which(name)
        if p:return p
    raise RuntimeError('未找到 Chromium。請用瀏覽器開啟 reports/*.html，列印另存 PDF。')


def chrome_run(extra, timeout=45):
    base=[find_browser(),'--headless','--no-sandbox','--disable-gpu','--disable-software-rasterizer',
          '--in-process-gpu','--no-first-run','--no-default-browser-check','--disable-extensions',
          '--disable-background-networking','--user-data-dir='+str(ROOT/'.chrome-profile-export')]
    done=subprocess.run(base+extra,capture_output=True,timeout=timeout)
    (ROOT/'results/chromium_last.log').write_bytes(done.stderr)
    if done.returncode:raise RuntimeError('Chromium failed; see results/chromium_last.log')
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-ai-log',action='store_true',help='另外匯出選填的 AI 對話紀錄')
    args=parser.parse_args()
    stems=['HW2_研究報告']
    if args.include_ai_log:
        stems.append('AI_協助與對話紀錄')
    for stem in stems:
        src=ROOT/'reports'/f'{stem}.html'
        dst=ROOT/'reports'/f'{stem}.pdf'
        if not src.exists():raise FileNotFoundError(src)
        chrome_run(['--no-pdf-header-footer','--print-to-pdf='+str(dst),'--virtual-time-budget=3000',src.as_uri()])
        if not dst.exists() or not dst.read_bytes().startswith(b'%PDF'):raise RuntimeError('PDF not created: '+str(dst))
        print(f'{dst.name}: {dst.stat().st_size:,} bytes')
    shutil.copyfile(ROOT/'reports/HW2_研究報告.pdf',ROOT/'web/dist/HW2_研究報告.pdf')


if __name__=='__main__':main()
