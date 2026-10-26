"""Execute this pure-Python notebook sequentially and retain real stdout outputs.

This runner does not emulate an IPython kernel. In plain Python, the plot cell
prints file paths; a normal Jupyter kernel renders the SVGs inline.
"""
from pathlib import Path
import json
import io
from contextlib import redirect_stdout
import os

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'HW2_Multiple_Linear_Regression.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    namespace={'__name__':'__main__'}
    count=0
    original=Path.cwd()
    os.chdir(ROOT)
    try:
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            count+=1
            output=io.StringIO()
            with redirect_stdout(output):
                exec(compile(''.join(cell['source']),str(path)+f':cell-{count}','exec'),namespace)
            cell['execution_count']=count
            text=output.getvalue()
            cell['outputs']=[dict(output_type='stream',name='stdout',text=text.splitlines(keepends=True))] if text else []
    finally:os.chdir(original)
    notebook['metadata']['execution_note']='Code cells executed sequentially by scripts/run_notebook.py in plain Python; SVG display falls back to file paths when IPython is absent.'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Executed {count} notebook code cells successfully; real stdout retained.')


if __name__=='__main__':main()
