"""Integration invariants and independent cross-runtime numeric verification."""
from pathlib import Path
import unittest
import sys
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict import predict_frame


class ResultsValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
        cls.models=json.loads((ROOT/'results/models.json').read_text(encoding='utf-8'))
        cls.df=pd.read_csv(ROOT/'data/flood.csv')
        cls.splits=pd.read_csv(ROOT/'results/split_manifest.csv')

    def test_split_is_partition(self):
        self.assertEqual(len(self.splits),len(self.df))
        self.assertEqual(self.splits.row_id.nunique(),50000)
        self.assertEqual(self.splits.groupby('split').size().to_dict(),{'train':30000,'validation':10000,'test':10000})
        self.assertEqual(set(self.splits.row_id),set(range(50000)))

    def test_preprocessing_uses_train_only(self):
        train=self.df.iloc[self.splits.loc[self.splits.split=='train','row_id']]
        for model in self.models.values():
            x=train[model['features']].to_numpy()
            np.testing.assert_allclose(model['mean'],x.mean(axis=0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(model['scale'],x.std(axis=0),rtol=1e-12,atol=1e-12)

    def test_selection_counts_and_validation_winner(self):
        self.assertEqual(len(self.s['models']),18)
        self.assertEqual(self.s['best']['key'],min(self.s['models'],key=lambda r:r['validation_RMSE'])['key'])
        for name,features in self.s['selected_features'].items():
            self.assertEqual(len(features),20 if name=='All20' else 14)
            self.assertTrue(set(features)<=set(self.s['features']))

    def test_exported_model_and_report_metrics_match(self):
        test=pd.read_csv(ROOT/'results/test_predictions.csv')
        frame=self.df.iloc[test.row_id]
        y=test.actual.to_numpy()
        for row in self.s['models']:
            p=predict_frame(frame,self.models[row['key']])
            np.testing.assert_allclose(p,test[row['key']],atol=1e-13,rtol=1e-12)
            rmse=np.sqrt(np.mean((y-p)**2))
            self.assertAlmostEqual(rmse,row['RMSE'],places=12)

    def test_mean_baseline_and_degenerate_diagnostics(self):
        self.assertLess(self.s['best']['RMSE'],self.s['baseline']['RMSE'])
        self.assertLess(self.s['deterministic_check']['max_absolute_difference'],1e-12)
        for check in self.s['diagnostics'][:4]:
            self.assertIsNone(check['p_value'])
        self.assertEqual(len(self.s['diagnostics_reduced14']),5)
        self.assertIn('無法確認',self.s['diagnostics_reduced14'][1]['conclusion'])

    def test_prediction_missing_and_invalid_input(self):
        m=self.models[self.s['best']['key']]
        frame=pd.read_csv(ROOT/'data/new_samples.csv')
        frame.loc[0,m['features'][0]]=np.nan
        self.assertTrue(np.isfinite(predict_frame(frame,m)).all())
        with self.assertRaises(ValueError):predict_frame(frame.drop(columns=m['features'][0]),m)
        frame.loc[0,m['features'][0]]=np.inf
        with self.assertRaises(ValueError):predict_frame(frame,m)

    def test_prediction_interval_calibrated_without_test_labels(self):
        info=self.s['prediction_interval']
        model=self.models[info['model']]
        val=self.df.iloc[self.splits.loc[self.splits.split=='validation','row_id']]
        residuals=np.sort(np.abs(val.FloodProbability.to_numpy()-predict_frame(val,model)))
        expected=residuals[int(np.ceil((len(val)+1)*.95))-1]
        self.assertAlmostEqual(info['radius'],expected,places=12)
        frame=pd.read_csv(ROOT/'results/prediction_intervals.csv')
        self.assertTrue((frame.lower95<=frame.predicted).all())
        self.assertTrue((frame.upper95>=frame.predicted).all())
        self.assertAlmostEqual(float(((frame.actual>=frame.lower95)&(frame.actual<=frame.upper95)).mean()),info['test_coverage'])
        self.assertGreater(info['test_coverage'],.9)
        self.assertLess(info['test_coverage'],1)

    def test_browser_math_matches_sklearn(self):
        node=shutil.which('node')
        if not node:self.skipTest('Node not installed')
        raw=subprocess.check_output([node,str(ROOT/'tests/test_browser_core.cjs'),'--reference'],cwd=ROOT)
        ref=json.loads(raw)
        X=np.array(ref['X']);y=np.array(ref['y'])
        for name,est in [('LinearRegression',LinearRegression()),('Ridge',Ridge(alpha=.3)),('Lasso',Lasso(alpha=.3,tol=1e-12,max_iter=10000))]:
            est.fit(X,y)
            np.testing.assert_allclose(ref['fitted'][name]['coef'],est.coef_,atol=1e-7,rtol=1e-7)
            self.assertAlmostEqual(ref['fitted'][name]['intercept'],est.intercept_,places=10)


if __name__=='__main__':unittest.main()
