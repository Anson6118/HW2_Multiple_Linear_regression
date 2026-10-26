"""Predict a local CSV with exported, training-fitted coefficients."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def predict_frame(frame, model):
    missing=set(model['features'])-set(frame.columns)
    if missing:
        raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    a=frame[model['features']].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if np.isinf(a).any():
        raise ValueError('Infinite feature values are not supported.')
    a=np.where(np.isnan(a),np.array(model['median']),a)
    return ((a-model['mean'])/model['scale'])@np.array(model['coefficients'])+model['intercept']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/predictions.csv')
    parser.add_argument('--model',default=None)
    args=parser.parse_args()
    models=json.loads((ROOT/'results/models.json').read_text(encoding='utf-8'))
    key=args.model or json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))['best']['key']
    if key not in models:
        parser.error('Unknown model: '+str(key))
    frame=pd.read_csv(args.csv)
    frame['prediction']=predict_frame(frame,models[key])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False)
    print(f'{len(frame)} predictions -> {args.output}')


if __name__=='__main__':
    main()
