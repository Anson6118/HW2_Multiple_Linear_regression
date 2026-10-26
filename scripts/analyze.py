"""Reproducible CRISP-DM regression experiment; no network required.

Run from repository root: python scripts/analyze.py
Feature and hyperparameter selection never sees the final test partition.
"""
from pathlib import Path
import json
import hashlib
import platform
import sys
import time
import html

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
WEB = ROOT / 'web' / 'dist'
SEED = 42
TARGET = 'FloodProbability'


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def metrics(y, pred, p):
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    mse = float(mean_squared_error(y, pred))
    r2 = float(r2_score(y, pred))
    mask = np.abs(y) > 1e-12
    return dict(MAE=float(mean_absolute_error(y, pred)), MSE=mse, RMSE=float(np.sqrt(mse)),
                R2=r2, Adjusted_R2=float(1-(1-r2)*(n-1)/(n-p-1)) if n > p+1 else None,
                MAPE=float(np.mean(np.abs((y[mask]-pred[mask])/y[mask]))*100) if mask.any() else None,
                MAPE_excluded_zeros=int((~mask).sum()))


def diagnostics(X, y, names):
    """OLS diagnostics on training rows in original CSV order.

    RESET: F-test adding fitted^2/fitted^3; BP: Koenker n*R² LM;
    normality: Jarque-Bera; independence: descriptive DW (not a proof);
    collinearity: VIF auxiliary regressions with intercept.
    """
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    fitted = A @ beta
    e = y-fitted
    rss = float(e @ e)
    fitted_z = (fitted-fitted.mean()) / fitted.std()
    B = np.column_stack([A, fitted_z**2, fitted_z**3])
    er = y-B @ np.linalg.lstsq(B, y, rcond=None)[0]
    rank_a, rank_b = np.linalg.matrix_rank(A), np.linalg.matrix_rank(B)
    df1, df2 = rank_b-rank_a, len(y)-rank_b
    reset_f = max(0.0, (rss-float(er@er))/df1/(float(er@er)/df2))
    e2 = e**2
    e2hat = A @ np.linalg.lstsq(A, e2, rcond=None)[0]
    bp_lm = len(y)*float(r2_score(e2, e2hat))
    jb = stats.jarque_bera(e)
    vifs = []
    for j, name in enumerate(names):
        aux = np.delete(A, j+1, axis=1)
        r2 = r2_score(X[:, j], aux @ np.linalg.lstsq(aux, X[:, j], rcond=None)[0])
        vifs.append(dict(feature=name, VIF=float(1/(1-r2))))
    checks = [
        dict(assumption='線性關係', test='Ramsey RESET (powers 2, 3)', statistic=reset_f,
             p_value=float(stats.f.sf(reset_f, df1, df2)), note='p < .05 表示有函數形式不足的證據；搭配殘差圖判讀。'),
        dict(assumption='殘差獨立性', test='Durbin–Watson (descriptive)', statistic=float(np.diff(e)@np.diff(e)/rss),
             p_value=None, note='依原始 CSV 列序計算。資料無時間、地區或群組資訊，無法驗證獨立性；接近 2 不能證明獨立。'),
        dict(assumption='等變異性', test='Koenker–Breusch–Pagan LM', statistic=bp_lm,
             p_value=float(stats.chi2.sf(bp_lm, rank_a-1)), note='p < .05 表示異質變異證據；未拒絕不等於證明等變異。'),
        dict(assumption='殘差常態性', test='Jarque–Bera', statistic=float(jb.statistic),
             p_value=float(jb.pvalue), note='檢查殘差，而非每個輸入特徵。大樣本對輕微偏離敏感；搭配 Q–Q 圖。'),
        dict(assumption='無嚴重多重共線性', test='Maximum VIF', statistic=max(v['VIF'] for v in vifs),
             p_value=None, note='VIF > 5 作為需關注的經驗門檻；共線性影響係數穩定性，不是因果證據。')]
    for c in checks:
        if c['p_value'] is not None:
            c['conclusion'] = '有違反假設的證據' if c['p_value'] < .05 else '未發現足夠違反證據'
        elif c['assumption'] == '殘差獨立性':
            c['conclusion'] = '缺少時間／群組資訊，無法確認'
        else:
            c['conclusion'] = '需關注共線性' if c['statistic'] > 5 else '未見嚴重共線性'
    if np.sqrt(rss/len(y)) < 1e-10 * max(1, np.std(y)):
        for c in checks[:4]:
            c['statistic'] = None
            c['p_value'] = None
            c['conclusion'] = '殘差近零，檢定退化，不作統計解讀'
            c['note'] = '此模型殘差僅有浮點運算誤差；改參考 SelectKBest14 的非退化殘差診斷。'
    return checks, vifs, fitted, e


def scatter_svg(path, x, y, title, xlabel, ylabel, diagonal=False, zero=False):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > 1500:
        idx = np.random.default_rng(SEED).choice(len(x), 1500, replace=False)
        x, y = x[idx], y[idx]
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    if diagonal:
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
    dx, dy = (xmax-xmin)*.05 or 1, (ymax-ymin)*.05 or 1
    xmin, xmax, ymin, ymax = xmin-dx, xmax+dx, ymin-dy, ymax+dy
    px = lambda v: 80+(v-xmin)/(xmax-xmin)*650
    py = lambda v: 410-(v-ymin)/(ymax-ymin)*335
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 490" role="img">',
         '<rect width="800" height="490" fill="#fff"/>',
         f'<text x="400" y="30" text-anchor="middle" font-family="Arial" font-size="20" fill="#102c39">{html.escape(title)}</text>']
    for t in np.linspace(0, 1, 6):
        xv, yv = xmin+t*(xmax-xmin), ymin+t*(ymax-ymin)
        s += [f'<path d="M80 {py(yv):.2f}H730 M{px(xv):.2f} 75V410" stroke="#e4edf0"/>',
              f'<text x="{px(xv):.2f}" y="433" text-anchor="middle" font-family="Arial" font-size="12">{xv:.3f}</text>',
              f'<text x="70" y="{py(yv)+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.3f}</text>']
    s += [f'<circle cx="{px(a):.2f}" cy="{py(b):.2f}" r="2.1" fill="#008d95" opacity=".35"/>' for a,b in zip(x,y)]
    if diagonal:
        s.append(f'<path d="M{px(xmin)} {py(xmin)}L{px(xmax)} {py(xmax)}" stroke="#e47a3f" stroke-width="2" stroke-dasharray="6 5"/>')
    if zero and ymin < 0 < ymax:
        s.append(f'<path d="M80 {py(0)}H730" stroke="#e47a3f" stroke-width="2"/>')
    s += [f'<text x="400" y="472" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(xlabel)}</text>',
          f'<text transform="translate(20 250) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(ylabel)}</text>', '</svg>']
    Path(path).write_text(''.join(s), encoding='utf-8')


def interval_svg(path,pred,actual,radius):
    """A fixed random test sample, ordered only for visual readability."""
    idx=np.random.default_rng(SEED).choice(len(pred),80,replace=False)
    idx=idx[np.argsort(pred[idx])]
    pred,actual=np.asarray(pred)[idx],np.asarray(actual)[idx]
    ymin=min((pred-radius).min(),actual.min())-.01
    ymax=max((pred+radius).max(),actual.max())+.01
    px=lambda i:75+i/(len(pred)-1)*650
    py=lambda v:365-(v-ymin)/(ymax-ymin)*295
    s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" role="img" aria-label="95 percent split conformal prediction intervals for OLS14">',
       '<rect width="800" height="450" fill="white"/>',
       '<text x="400" y="30" text-anchor="middle" font-family="Arial" font-size="18" fill="#173b43">OLS14: 95% prediction intervals (test)</text>']
    for tick in np.linspace(ymin,ymax,6):
        s.append(f'<path d="M75 {py(tick)}H725" stroke="#e4ece7"/><text x="65" y="{py(tick)+4}" text-anchor="end" font-family="Arial" font-size="11">{tick:.3f}</text>')
    upper=[f'{px(i):.2f},{py(v+radius):.2f}' for i,v in enumerate(pred)]
    lower=[f'{px(i):.2f},{py(v-radius):.2f}' for i,v in reversed(list(enumerate(pred)))]
    s.append('<polygon points="'+' '.join(upper+lower)+'" fill="#e9a267" opacity=".28"/>')
    s.append('<polyline points="'+' '.join(f'{px(i):.2f},{py(v):.2f}' for i,v in enumerate(pred))+'" stroke="#173b43" stroke-width="2" fill="none"/>')
    s.extend(f'<circle cx="{px(i):.2f}" cy="{py(v):.2f}" r="3" fill="#008a88"/>' for i,v in enumerate(actual))
    s+=['<text x="400" y="399" text-anchor="middle" font-family="Arial" font-size="12">80 fixed test samples, ordered by point prediction</text>',
        '<text x="110" y="430" font-family="Arial" font-size="11" fill="#008a88">Teal dots: actual</text>',
        '<text x="300" y="430" font-family="Arial" font-size="11" fill="#173b43">Dark line: predicted</text>',
        '<text x="500" y="430" font-family="Arial" font-size="11" fill="#b46f36">Orange band: 95% interval</text></svg>']
    Path(path).write_text(''.join(s),encoding='utf-8')


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    path = ROOT/'data/flood.csv'
    df = pd.read_csv(path)
    names = [c for c in df.columns if c != TARGET and c.lower() not in ['id', 'index']]
    assert 13 < len(names) < 40 and len(names) == 20
    assert not df.duplicated().any(), 'Duplicate rows need grouping before splitting.'
    assert not df[TARGET].isna().any()
    X, y = df[names], df[TARGET]
    train_i, remainder_i = train_test_split(np.arange(len(df)), test_size=.4, random_state=SEED)
    val_i, test_i = train_test_split(remainder_i, test_size=.5, random_state=SEED)
    Xtr, Xval, Xtest = [X.iloc[i] for i in [train_i, val_i, test_i]]
    ytr, yval, ytest = [y.iloc[i] for i in [train_i, val_i, test_i]]
    pre = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    ztr = pre.fit_transform(Xtr)
    k = 14
    selectors, rankings = {'All20': np.arange(len(names))}, {}
    sk = SelectKBest(f_regression, k=k).fit(ztr, ytr)
    selectors['SelectKBest14'] = sk.get_support(indices=True)
    rankings['SelectKBest'] = sk.scores_
    rfe = RFE(Ridge(alpha=1), n_features_to_select=k).fit(ztr, ytr)
    selectors['RFE14'] = rfe.get_support(indices=True)
    rankings['RFE'] = -rfe.ranking_
    # A reproducible training-only subsample keeps backward SFS affordable.
    rng = np.random.default_rng(SEED)
    sfs_i = rng.choice(len(Xtr), min(5000, len(Xtr)), replace=False)
    sfs_est = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1))
    sfs = SequentialFeatureSelector(sfs_est, n_features_to_select=k, direction='backward',
                                    scoring='neg_root_mean_squared_error',
                                    cv=KFold(3, shuffle=True, random_state=SEED), n_jobs=1)
    print('Fitting backward Sequential Feature Selection (training-only n=5000)...', flush=True)
    sfs.fit(Xtr.iloc[sfs_i], ytr.iloc[sfs_i])
    selectors['SFS14'] = sfs.get_support(indices=True)
    lasso_rank = Lasso(alpha=.001, max_iter=10000).fit(ztr, ytr)
    rankings['Lasso'] = np.abs(lasso_rank.coef_)
    selectors['Lasso14'] = np.sort(np.argsort(-rankings['Lasso'], kind='stable')[:k])
    perm_a, perm_b = train_test_split(np.arange(len(Xtr)), test_size=.25, random_state=SEED)
    perm_model = clone(sfs_est).fit(Xtr.iloc[perm_a], ytr.iloc[perm_a])
    perm = permutation_importance(perm_model, Xtr.iloc[perm_b], ytr.iloc[perm_b],
                                 n_repeats=5, random_state=SEED, scoring='neg_root_mean_squared_error')
    rankings['Permutation'] = perm.importances_mean
    selectors['Permutation14'] = np.sort(np.argsort(-perm.importances_mean, kind='stable')[:k])
    rows, candidates, predictions, validations = [], {}, {}, []
    grids = {'LinearRegression': [None], 'Ridge': [.01, .1, 1, 10, 100], 'Lasso': [.000001,.00001,.0001,.001,.01]}
    for method, idx in selectors.items():
        cols = [names[i] for i in idx]
        for model_name, alphas in grids.items():
            choices=[]
            for alpha in alphas:
                est = LinearRegression() if model_name == 'LinearRegression' else (
                    Ridge(alpha=alpha) if model_name == 'Ridge' else Lasso(alpha=alpha, max_iter=10000))
                pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)
                pipe.fit(Xtr[cols], ytr)
                val_rmse = float(np.sqrt(mean_squared_error(yval, pipe.predict(Xval[cols]))))
                choices.append((val_rmse, alpha, pipe))
                validations.append(dict(selector=method, model=model_name, alpha=alpha, validation_RMSE=val_rmse))
            val_rmse, alpha, fitted = min(choices, key=lambda t: t[0])
            key = method+' / '+model_name
            candidates[key] = (cols, fitted)
            # All test scores are descriptive comparisons; only validation selects the winner.
            pred = fitted.predict(Xtest[cols])
            predictions[key] = pred
            rows.append(dict(key=key, selector=method, model=model_name, features=len(cols), alpha=alpha,
                             validation_RMSE=val_rmse, **metrics(ytest,pred,len(cols))))
    best = min(rows, key=lambda row: row['validation_RMSE'])
    # Fixed training-selected OLS14; no alpha/model selection using calibration labels.
    # Reuse validation as calibration only for this explicitly fixed auxiliary model.
    interval_key='SelectKBest14 / LinearRegression'
    interval_cols,interval_model=candidates[interval_key]
    cal_scores=np.abs(yval.to_numpy()-interval_model.predict(Xval[interval_cols]))
    confidence=.95
    rank=int(np.ceil((len(cal_scores)+1)*confidence))
    radius=float(np.partition(cal_scores,min(rank,len(cal_scores))-1)[min(rank,len(cal_scores))-1])
    interval_pred=predictions[interval_key]
    lower,upper=interval_pred-radius,interval_pred+radius
    coverage=float(np.mean((ytest.to_numpy()>=lower)&(ytest.to_numpy()<=upper)))
    intervals=dict(model=interval_key,method='Split conformal absolute-residual prediction interval',
                   confidence=confidence,calibration_count=len(cal_scores),rank=rank,radius=radius,
                   test_coverage=coverage,mean_width=float(2*radius),
                   note='固定使用只由 train 決定的 SelectKBest14 / OLS，沒有用 calibration labels 挑此區間模型或 alpha。外層 validation 同時是此固定輔助模型的 calibration；test 未用來調整寬度。名目邊際涵蓋率依賴交換性假設，不保證每個條件或真實地區的涵蓋率。')
    dump(OUT/'prediction_interval_summary.json',intervals)
    pd.DataFrame(dict(row_id=test_i,actual=ytest.to_numpy(),predicted=interval_pred,lower95=lower,upper95=upper)).to_csv(OUT/'prediction_intervals.csv',index=False)
    interval_svg(OUT/'prediction_intervals.svg',interval_pred,ytest.to_numpy(),radius)
    cols, fitted = candidates[best['key']]
    pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(validations).to_csv(OUT/'validation_search.csv',index=False)
    split = pd.DataFrame(dict(row_id=np.arange(len(df)),split='train'))
    split.loc[val_i,'split']='validation'
    split.loc[test_i,'split']='test'
    split.to_csv(OUT/'split_manifest.csv',index=False)
    predictions_df = pd.DataFrame({'row_id':test_i,'actual':ytest.to_numpy(),
                                   **{key:pred for key,pred in predictions.items()}})
    predictions_df.to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame({name:[names[i] for i in idx] + ['']*(20-len(idx)) for name,idx in selectors.items()}).to_csv(OUT/'selected_features.csv',index=False)
    pd.DataFrame(rankings,index=names).rename_axis('feature').to_csv(OUT/'feature_rankings.csv')
    Xtr.corr().to_csv(OUT/'feature_correlations.csv')
    Xtr.corrwith(ytr).rename('target_correlation').to_csv(OUT/'target_correlations.csv')
    df.describe().T.to_csv(OUT/'data_description.csv')
    order=np.argsort(train_i)
    checks,vifs,fit_train,resid_train=diagnostics(ztr[order],ytr.to_numpy()[order],names)
    reduced_idx=selectors['SelectKBest14']
    reduced_checks,_,reduced_fit,reduced_resid=diagnostics(ztr[order][:,reduced_idx],ytr.to_numpy()[order],[names[i] for i in reduced_idx])
    pd.DataFrame(vifs).to_csv(OUT/'vif.csv',index=False)
    dump(OUT/'assumptions.json',checks)
    dump(OUT/'assumptions_reduced14.json',reduced_checks)
    exports={}
    for key,(features,pipe) in candidates.items():
        imputer,scaler,est=pipe.steps[0][1],pipe.steps[1][1],pipe.steps[2][1]
        exports[key]=dict(features=features,median=imputer.statistics_.tolist(),
                          mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                          coefficients=est.coef_.tolist(),intercept=float(est.intercept_))
    dump(OUT/'models.json',exports)
    coeff=fitted.steps[-1][1].coef_/fitted.steps[1][1].scale_
    intercept=float(fitted.steps[-1][1].intercept_-np.dot(coeff,fitted.steps[1][1].mean_))
    pd.DataFrame({'feature':cols,'coefficient_original_unit':coeff,
                  'coefficient_standardized':fitted.steps[-1][1].coef_}).to_csv(OUT/'best_coefficients.csv',index=False)
    new=Xtr.median().to_frame().T
    new=pd.concat([new,new,new],ignore_index=True)
    new.loc[1,'MonsoonIntensity']=Xtr['MonsoonIntensity'].quantile(.9)
    new.loc[2,'MonsoonIntensity']=Xtr['MonsoonIntensity'].quantile(.1)
    new.to_csv(ROOT/'data/new_samples.csv',index=False)
    new['prediction']=fitted.predict(new[cols])
    new['prediction_OLS14']=interval_model.predict(new[interval_cols])
    new['lower95_OLS14']=new['prediction_OLS14']-radius
    new['upper95_OLS14']=new['prediction_OLS14']+radius
    new.to_csv(OUT/'new_predictions.csv',index=False)
    plot_idx=np.random.default_rng(SEED).choice(len(ytest),1000,replace=False)
    pred=predictions[best['key']]
    scatter_svg(OUT/'actual_vs_predicted.svg',ytest,pred,'Actual vs Predicted (test)','Actual FloodProbability','Predicted FloodProbability',diagonal=True)
    scatter_svg(OUT/'residuals.svg',pred,ytest.to_numpy()-pred,'Residuals vs Predicted (test)','Predicted FloodProbability','Residual (actual - predicted)',zero=True)
    q=stats.probplot(reduced_resid,dist='norm',fit=False)
    scatter_svg(OUT/'qq_plot.svg',q[0],np.asarray(q[1])/reduced_resid.std(),'OLS14 residual Q-Q (train)','Theoretical normal quantile','Standardized residual',diagonal=True)
    scatter_svg(OUT/'residual_order.svg',np.sort(train_i),reduced_resid,'OLS14 residual by original row (train)','Original CSV row','Residual',zero=True)
    scatter_svg(OUT/'residuals_reduced14.svg',reduced_fit,reduced_resid,'OLS14 residuals vs fitted (train)','Fitted FloodProbability','Residual',zero=True)
    summary=dict(dataset='Kaggle Flood Prediction Dataset (flood.csv)',target=TARGET,rows=len(df),
                 features=names,feature_count=len(names),missing=int(df.isna().sum().sum()),duplicates=int(df.duplicated().sum()),
                 split=dict(train=len(train_i),validation=len(val_i),test=len(test_i),seed=SEED),
                 target_range=[float(y.min()),float(y.max())],best=best,models=rows,
                 eda=dict(target_mean=float(y.mean()),target_std=float(y.std()),
                          target_quartiles={str(q):float(y.quantile(q)) for q in [.25,.5,.75]},
                          train_target_correlations=Xtr.corrwith(ytr).to_dict(),
                          train_max_feature_correlation=float(Xtr.corr().where(~np.eye(len(names),dtype=bool)).abs().max().max())),
                 prediction_interval=intervals,
                 baseline=metrics(ytest,np.repeat(ytr.mean(),len(ytest)),0),
                 selected_features={m:[names[i] for i in idx] for m,idx in selectors.items()},
                 diagnostics=checks,diagnostics_reduced14=reduced_checks,vif=vifs,original_unit_equation=dict(intercept=intercept,coefficients=dict(zip(cols,coeff.tolist()))),
                 deterministic_check=dict(formula='FloodProbability = 0.005 × sum(20 features)',
                                          max_absolute_difference=float(np.abs(y-.005*X.sum(axis=1)).max()),
                                          note='建模完成後才檢查的描述性資料特性；未據此修改模型或選擇候選模型。'),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 source=dict(kaggle='https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset',
                             mirror='https://github.com/SumeetSantoshjunnarkar/Floodsense/blob/f40ca4475573f3be79a71fa076f677cca3893ea9/archive/flood.csv',
                             git_blob_sha='2c01d36b3fd311430e557548759c551367dd6ad6',
                             caveat='同名 flood.csv 的公開鏡像；本機檔案雜湊已保存。未直接以 Kaggle API 核對原始檔位元組。合成／未驗證地理時序來源的教學資料，不代表真實防災風險。'),
                 methodology=dict(selection='所有 selector 只接觸 30,000 訓練列；5,000 列訓練子樣本做 backward SFS（三折，fold 內 preprocessing）；Permutation 使用訓練集內 75/25 子切分。',
                                  tuning='用 10,000 列 validation 決定模型與 alpha；未用 test 選模型；未在 train+validation 重訓。',
                                  lasso='Lasso14 為標準化後 alpha=.001 的絕對係數前 14 名；非聲稱只有 14 個非零係數。',
                                  counts='原始特徵 20；比較 14 與 20。資料不足 25/30 個原始特徵，不刻意造特徵湊數。',
                                  adjusted_r2='1-(1-R²)(n-1)/(n-p-1)，test n=10000，p=欄位數。Ridge/Lasso 的 p 並非有效自由度，僅作描述性比較。'),
                 versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                 elapsed_seconds=round(time.time()-started,2))
    dump(OUT/'summary.json',summary)
    # Compact browser data; real fitted models are exported without approximating coefficients.
    scatter=[dict(actual=float(ytest.iloc[i]),predicted=float(pred[i])) for i in plot_idx]
    bundle=dict(summary=summary,models=exports,scatter=scatter,
                test_sample=[dict(features=Xtest.iloc[i].to_dict(),actual=float(ytest.iloc[i])) for i in plot_idx],
                sample_rows=Xtr.sample(10,random_state=SEED).to_dict(orient='records'),
                training_ranges={c:[float(Xtr[c].min()),float(Xtr[c].max())] for c in names})
    (WEB/'results.js').write_text('window.HW2_RESULTS='+json.dumps(bundle,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    df.sample(2000,random_state=SEED).to_csv(WEB/'flood_sample.csv',index=False)
    import shutil
    shutil.copyfile(ROOT/'data/50_Startups.csv',WEB/'50_Startups.csv')
    shutil.copyfile(OUT/'prediction_intervals.svg',WEB/'prediction_intervals.svg')
    shutil.copyfile(OUT/'prediction_intervals.csv',WEB/'prediction_intervals.csv')
    (WEB/'examples.js').write_text('window.HW2_EXAMPLES='+json.dumps(dict(
        flood=(WEB/'flood_sample.csv').read_text(encoding='utf-8'),
        startups=(WEB/'50_Startups.csv').read_text(encoding='utf-8')),ensure_ascii=False)+';\n',encoding='utf-8')
    print(json.dumps(dict(best=best,baseline=summary['baseline'],diagnostics=checks,elapsed=summary['elapsed_seconds']),ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
