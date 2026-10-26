"""Build the Chinese report, notebook, data provenance and print-ready AI log."""
from pathlib import Path
import json
import html
import shutil
import datetime

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
WEB=ROOT/'web/dist'


def f(v):
    if v is None:return '—'
    return f'{v:.3e}' if v!=0 and abs(v)<.0001 else f'{v:,.6f}'


def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


CSS='''
@page{size:A4;margin:16mm 14mm}*{box-sizing:border-box}body{color:#173b43;background:#f5f7f3;font:14px/1.85 "Microsoft JhengHei","PingFang TC",Arial,sans-serif;margin:0}main{max-width:1050px;margin:30px auto;background:white;padding:50px}h1{font-size:34px;line-height:1.4;margin:10px 0 20px}h2{font-size:23px;border-bottom:2px solid #dbe9e3;padding-bottom:8px;margin-top:40px}h3{font-size:17px;margin-top:25px}p{margin:12px 0}a{color:#007a79;overflow-wrap:anywhere}code,pre{font-family:Consolas,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#eef5f0;border-radius:8px}.eyebrow{letter-spacing:2px;font-size:11px;color:#008a88}.lead{font-size:17px}.note{border-left:4px solid #db965b;background:#fbf2e7;padding:16px 20px}.good{border-left:4px solid #008a88;background:#edf6ee;padding:16px 20px}.small{font-size:11px;color:#687d79}.table-wrap{overflow:auto;margin:15px 0}table{width:100%;border-collapse:collapse;font-size:10px}th,td{padding:8px;border:1px solid #dce6df;text-align:left;vertical-align:top}th{background:#edf5ef}img,svg{display:block;width:100%;height:auto;margin:16px auto;max-height:500px;object-fit:contain}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.columns img{margin:0}li{margin-bottom:6px}.nav{display:flex;gap:20px;font-size:12px}.transcript{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.8 "Microsoft JhengHei",sans-serif}.entry{padding:15px 0;border-bottom:1px solid #e1e8e2;break-inside:avoid}.role{font-size:11px;color:#008a88}.hash{font-family:Consolas;font-size:10px;overflow-wrap:anywhere}@media print{body{background:white;font-size:10.5px}main{margin:0;padding:0;max-width:none}h1{font-size:26px}h2{font-size:18px;break-after:avoid}h3{font-size:14px;break-after:avoid}p{orphans:3;widows:3}pre{font-size:9px}table{font-size:8px}th,td{padding:5px}thead{display:table-header-group}tr{break-inside:avoid}img,svg{break-inside:avoid;max-height:230mm}.nav,.screen-only{display:none}.columns{display:block}.columns img{max-height:95mm}a{color:inherit}.note,.good{break-inside:avoid}.table-wrap{overflow:visible}}@media(max-width:700px){main{padding:24px;margin:0}h1{font-size:27px}.columns{grid-template-columns:1fr}}
'''


def document(title,body):
    return '<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+CSS+'</style></head><body><main>'+body+'</main></body></html>'


def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    b=s['best']
    names=s['features']
    translations=['季風強度','地形排水','河流管理','森林砍伐','都市化','氣候變遷','水壩品質','淤積','農業活動','侵占開發','防災準備不足','排水系統','海岸脆弱度','山崩','集水區','基礎設施劣化','人口分數','濕地流失','規劃不足','政治因素']
    models=table(['特徵 / 模型','p','alpha','驗證 RMSE','測試 MAE','MSE','RMSE','R²','Adj. R²','MAPE %'],[
        [html.escape(r['key']),r['features'],r['alpha'] if r['alpha'] is not None else '—']+[f(r[k]) for k in ['validation_RMSE','MAE','MSE','RMSE','R2','Adjusted_R2','MAPE']] for r in s['models']])
    checks=table(['假設','檢查方式','統計量','p-value','判讀'],[[r['assumption'],r['test'],f(r['statistic']),f(r['p_value']),r['conclusion']] for r in s['diagnostics_reduced14']])
    pi=s['prediction_interval']
    features=table(['特徵','中文對照','型態'],[[html.escape(n),t,'數值分數；無實體單位'] for n,t in zip(names,translations)])
    eda=s['eda']
    correlations=table(['訓練集目標相關係數前 5 名','Pearson r'],[[name,f(corr)] for name,corr in sorted(eda['train_target_correlations'].items(),key=lambda item:abs(item[1]),reverse=True)[:5]])
    subsets=table(['方法','入選特徵'],[[m,'、'.join(cols)] for m,cols in s['selected_features'].items()])
    refs=[
      ('Kaggle Flood Prediction Dataset','https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset'),
      ('實際讀取檔案的公開鏡像（固定 commit）',s['source']['mirror']),
      ('scikit-learn：Feature selection','https://scikit-learn.org/stable/modules/feature_selection.html'),
      ('scikit-learn：Common pitfalls / data leakage','https://scikit-learn.org/stable/common_pitfalls.html'),
      ('scikit-learn：Linear models','https://scikit-learn.org/stable/modules/linear_model.html'),
      ('scikit-learn：Permutation importance','https://scikit-learn.org/stable/modules/permutation_importance.html'),
      ('statsmodels：Regression diagnostics','https://www.statsmodels.org/stable/diagnostic.html'),
      ('Angelopoulos & Bates：A Gentle Introduction to Conformal Prediction','https://arxiv.org/abs/2107.07511'),
      ('加分示範：50 Startups CSV','https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv')]
    # Inline SVG makes the report self-contained, including offline printing.
    svg=lambda name:(ROOT/'results'/name).read_text(encoding='utf-8')
    body=f'''
    <div class="nav"><a href="index.html">← 互動分析網頁</a><a href="HW2_研究報告.pdf">下載 PDF</a><a href="#ai">選填項目說明</a></div>
    <p class="eyebrow">AIoT · HW 2 · CRISP-DM + FEATURE SELECTION</p>
    <h1>洪水機率預測：<br>多元線性迴歸完整分析</h1>
    <p class="lead">50,000 筆資料 × 20 個原始特徵 × 18 組模型比較</p>
    <p class="small">日期：2026-10-22 ｜程式：Python / scikit-learn ｜學號、姓名：由本人填寫</p>
    <div class="good"><b>研究結果：</b>依驗證集選出 {html.escape(b['key'])}，獨立測試集 RMSE = {f(b['RMSE'])}、R² = {f(b['R2'])}。這份資料的目標可由二十個輸入的線性總和精確還原，因此結果主要反映資料的生成規則。</div>
    <div class="note"><b>完成範圍：</b>分析、程式、互動網頁與本機文件已實作；GitHub／網站繳交連結尚待確認，姓名、學號待填。依老師說明，整個第 4 項 AI 協助要求為選填，不列入待完成目標。</div>

    <h2>1. Business Understanding：商業理解</h2>
    <p>本研究建立可解釋的洪水機率分數預測流程。輸入環境、人口與基礎設施相關數值，預測連續目標 <code>FloodProbability</code>；這是迴歸任務，不將目標二元化。假想使用情境為教學性的風險篩選與模型比較，成功條件是優於僅預測訓練平均值的基準，並能重現特徵選擇、評估與新樣本預測。</p>
    <p>原始資料來源未提供可驗證的地點、時間、實體量測單位或事件觀測機制，因此不能直接投入真實防災決策。若未來要實際部署，應加入獨立地區／期間驗證與來源查證，而非用本資料的 R² 推論實務成效。</p>

    <h2>2. Data Understanding：資料理解</h2>
    <p>選用 Kaggle <a href="{s['source']['kaggle']}">Flood Prediction Dataset</a> 中的 <code>flood.csv</code> 同名公開鏡像。這份檔案有 {s['rows']:,} 列、21 欄；扣除目標後原始輸入為 20 個，符合 <b>13 &lt; 特徵數 &lt; 40</b>。這不是 1,117,957 列的競賽 <code>train.csv</code>，兩者不可混稱。</p>
    <p>缺失值 {s['missing']}、完全重複列 {s['duplicates']}。目標範圍為 {s['target_range'][0]}–{s['target_range'][1]}。原始檔來自<a href="{s['source']['mirror']}">固定 GitHub commit 的鏡像</a>，已記錄 Git blob 與本機 SHA-256；沒有宣稱曾以 Kaggle API 下載並核對位元組。</p>
    <p class="hash">本機 data/flood.csv SHA-256：{s['sha256']}</p>
    {features}
    <p>表中中文為欄名對照，不自行賦予分數方向。尤其 RiverManagement、DrainageSystems 等名稱不代表實際量表為「品質越高越安全」；本檔的統計關係應以實測結果解讀。訓練集特徵相關矩陣、目標相關係數、完整描述統計分別保存在 <code>feature_correlations.csv</code>、<code>target_correlations.csv</code>、<code>data_description.csv</code>。</p>
    <p>全資料目標平均 {f(eda['target_mean'])}、樣本標準差 {f(eda['target_std'])}；Q1={f(eda['target_quartiles']['0.25'])}、中位數={f(eda['target_quartiles']['0.5'])}、Q3={f(eda['target_quartiles']['0.75'])}。訓練集任兩輸入特徵的最大絕對 Pearson r 為 {f(eda['train_max_feature_correlation'])}，輸入彼此相關很低。以下呈現訓練資料的目標相關係數前五名；最終仍依各 selector 的規則比較。</p>
    {correlations}
    <div class="note"><b>資料生成規則：</b>完成模型比較後，描述性檢查發現 <code>FloodProbability = 0.005 × sum(20 features)</code>，全資料最大絕對差僅 {f(s['deterministic_check']['max_absolute_difference'])}。這是一項事後資料特性觀察，未用它重新挑選或修改模型。完美分數不代表真實世界中的完美預測。</div>

    <h2>3. Data Preparation：資料準備</h2>
    <p>固定 <code>random_state=42</code>，先隨機切成訓練 30,000 列、驗證 10,000 列、測試 10,000 列。對每個候選模型，<code>SimpleImputer(strategy="median") → StandardScaler → estimator</code> 的參數只從訓練資料估計。資料沒有缺失值，補值步驟仍保留以支援未來輸入；不補目標值。完整切分列索引保存在 <code>split_manifest.csv</code>。</p>
    <p>五種 selector 皆不接觸外層驗證／測試目標，先在訓練資料建立 14 特徵子集；一般模型的 alpha 和最終勝出組合由外層驗證 RMSE 決定。測試分數僅為最終描述性比較，不根據測試表格重新選勝出模型。資料沒有類別特徵，主實驗不做 one-hot；額外 CSV 實驗室對類別欄採訓練資料建立編碼。</p>
    <p>沒有以全資料刪除離群值或依目標進行特徵工程，避免人為改善測試成績。新輸入超過訓練範圍時屬外插；網頁顯示提示，仍保留模型原始輸出。</p>

    <h2>4. Modeling：建模與特徵選擇</h2>
    <p>依 <a href="https://scikit-learn.org/stable/modules/linear_model.html">scikit-learn 線性模型文件</a>，一般線性迴歸最小化平方殘差；Ridge 加入 L2 係數懲罰；Lasso 使用 L1 懲罰，可能產生零係數。兩種正則化方法都先標準化。alpha 的選擇範圍：Ridge = 0.01、0.1、1、10、100；Lasso = 10⁻⁶、10⁻⁵、10⁻⁴、10⁻³、10⁻²。</p>
    <p>以完整 20 特徵作為基準，另比較五種 14 特徵子集。只有 20 個原始特徵，因此不勉強產生 25 或 30 特徵的版本。各 selector 的方法依據見 <a href="https://scikit-learn.org/stable/modules/feature_selection.html">Feature selection</a>，實作設定如下。</p>
    <ul>
    <li><b>SelectKBest：</b>以 <code>f_regression</code> 單變量 F 分數選出前 14 欄。</li>
    <li><b>RFE：</b>使用 Ridge(alpha=1) 逐步遞迴淘汰，保留 14 欄。</li>
    <li><b>Sequential Feature Selection：</b>backward SFS，在固定抽樣的 5,000 筆訓練列上做 3 折 CV，以負 RMSE 評分；每折內各自補值與標準化。採子樣本是計算成本取捨，結果不代表在全部訓練列上最優。</li>
    <li><b>Lasso 排名：</b>在標準化訓練資料上擬合 alpha=.001，再取絕對係數前 14 欄；這是 top-k 排名，不能聲稱 Lasso 恰好得到 14 個非零係數。</li>
    <li><b>Permutation importance：</b>僅將外層訓練集內部切為 75% 擬合／25% 排名，以 Ridge(alpha=1)、5 次打亂造成的 RMSE 變化選前 14 欄。外層驗證和測試不參與排名。此重要性是模型與資料相關的預測指標，不是因果效果。</li></ul>
    {subsets}
    <p>6 種特徵組合 × 3 種模型 = 18 組。模型皆僅擬合外層訓練資料，沒有在選定參數後以 train+validation 重訓；匯出模型與報告評估的模型完全相同。</p>

    <h2>5. Evaluation：模型評估</h2>
    <p>MAE = mean(|y−ŷ|)；MSE = mean((y−ŷ)²)；RMSE = √MSE；R² = 1−SSE/SST；Adjusted R² = 1−(1−R²)(n−1)/(n−p−1)；MAPE = mean(|(y−ŷ)/y|)×100%。MAPE 排除 y=0 的列並記錄數量，本測試集無須排除。Adjusted R² 此處用測試樣本 n=10,000 與欄位數 p；Ridge/Lasso 的有效自由度不等於 p，因此僅作描述比較。</p>
    {models}
    <p>僅預測訓練平均值的基準測試 RMSE = {f(s['baseline']['RMSE'])}、MAE = {f(s['baseline']['MAE'])}、R² = {f(s['baseline']['R2'])}。完整特徵一般線性模型幾乎精確還原目標；14 特徵線性模型 R² 約 0.70，說明本資料的二十個因子都攜帶資訊，刪除欄位會損失預測能力。正則化在此資料上沒有帶來實質改善，過強懲罰反而產生偏差。</p>
    <h3>實際值與預測值、殘差圖</h3>
    <p class="small">最佳模型完整測試指標使用 10,000 筆；圖為固定最多 1,500 筆視覺抽樣。殘差 = 實際 − 預測。全特徵殘差只有浮點誤差，縱軸放大不代表實質預測誤差。</p>
    <div class="columns">{svg('actual_vs_predicted.svg')}{svg('residuals.svg')}</div>
    <h3>95% 預測區間：新觀測值的不確定性</h3>
    <p>為滿足題目「預測圖加上信賴區間或預測區間」要求，另外固定使用 <b>SelectKBest14 / LinearRegression</b> 產生 split conformal 預測區間，方法依據 <a href="https://arxiv.org/abs/2107.07511">Angelopoulos 與 Bates 的 conformal prediction 教程</a>。其特徵選擇與模型只由訓練資料決定，此輔助模型不使用驗證目標挑選模型或 alpha；將 10,000 筆驗證列另作它的校準資料。以校準絕對殘差排序，取第 ceil((n_cal+1)×0.95) = {pi['rank']} 小的值 q = {f(pi['radius'])}，對每筆測試預測建立 [ŷ−q, ŷ+q]。測試資料沒有用來調整區間寬度。</p>
    <p>10,000 筆獨立測試的實際涵蓋率為 <b>{pi['test_coverage']:.2%}</b>，平均區間寬度 {f(pi['mean_width'])}。圖中顯示固定抽樣 80 筆測試觀測，依點預測排序，橘色帶為預測區間、綠點為實際值。這是新觀測值的預測區間，不是條件平均的信賴區間；名目邊際涵蓋性依賴校準與未來觀測的交換性，不保證每個輸入條件或真實地區的涵蓋率。完整 20 特徵模型區間會縮至浮點誤差，因此不把該退化區間當成真實預測確定性。</p>
    {svg('prediction_intervals.svg')}
    <h3>五項迴歸假設檢查</h3>
    <p>依 <a href="https://www.statsmodels.org/stable/diagnostic.html">迴歸診斷文件</a>的定義檢查線性、獨立性、等變異、殘差常態與共線性。主實驗使用 NumPy/SciPy 實作輔助迴歸和統計分布，沒有安裝 statsmodels；各檢定公式在程式中註明。完整 20 特徵 OLS 的前四項檢定因殘差接近零而退化，不解讀浮點誤差產生的 p-value；完整模型最大 VIF = {max(v['VIF'] for v in s['vif']):.6f}。</p>
    <p>以下改用 <b>SelectKBest14 / LinearRegression 的訓練殘差</b>診斷（n=30,000），並保留一般條件與解讀限制：</p>
    {checks}
    <ul><li>線性：RESET 比較原模型與加入擬合值二次／三次項的擴充模型，F 檢定未發現函數形式不足的充分證據。</li>
    <li>獨立性：依原始 CSV 列序排列訓練殘差，DW 接近 2；因沒有實際時間或群組資訊，仍<b>無法確認觀測獨立性</b>。</li>
    <li>等變異：以殘差平方對設計矩陣做輔助迴歸，Koenker–BP 的 LM=n×R²，卡方自由度為設計矩陣秩減 1；未拒絕等變異。</li>
    <li>常態性：Jarque–Bera 拒絕殘差常態，需配合 Q–Q 圖判讀，且大樣本對微小偏離敏感。這不會使樣本外預測指標失效；若要傳統係數推論或區間，須另評估穩健標準誤／bootstrap。</li>
    <li>共線性：逐一以其他變項預測該特徵，VIF=1/(1−R²)，最大值接近 1，未見嚴重共線性。VIF&gt;5 僅為關注門檻。</li></ul>
    <div class="columns">{svg('residuals_reduced14.svg')}{svg('qq_plot.svg')}</div>
    {svg('residual_order.svg')}

    <h2>6. Deployment：預測與展示</h2>
    <p>模型包含訓練中位數、平均值、標準差、係數與截距，以 JSON 匯出；網頁直接使用同一組參數，無須安裝套件即可執行單筆預測。Python 批次預測指令：</p>
    <pre>python scripts/predict.py data/new_samples.csv --output results/predictions.csv</pre>
    <p>本模型的原始單位方程式約為 ŷ = 0 + 0.005×MonsoonIntensity + … + 0.005×PoliticalFactors。單個特徵增加 1，其餘固定時，模型輸出約增加 0.005；這是本資料的模型關係，不是現實中的因果結論。三筆新樣本輸入與預測保存在 <code>data/new_samples.csv</code> 和 <code>results/new_predictions.csv</code>。</p>
    <p>互動頁提供模型選擇、六指標、散佈圖、殘差圖、特徵名單及新樣本輸入。互動分析採 <b>單一分析區與資料集下拉選單</b>，可選 Flood、50 Startups 或自選資料集；選自選資料集時顯示 <b>CSV 檔案／網址匯入</b>。網址需為公開 HTTP(S) CSV，且來源伺服器允許跨來源存取；讀取失敗可改用下載後的檔案。類別欄在訓練集建立 one-hot，省略首類避免虛擬變項共線性；未知或缺失類別回到基準編碼。瀏覽器端最多 50,000 列、39 個原始輸入與 100 個編碼欄位。</p>
    <p>50 Startups 原始輸入只有 4 欄，僅作為題目允許的任意 CSV 匯入範例，不拿它充當符合 14–39 特徵條件的主資料。瀏覽器以固定 seed=42 重新分割 60/20/20；alpha 為使用者設定，沒有自動調參。瀏覽器實驗與正式 Python 結果獨立。</p>

    <h2 id="ai">7. 選填項目說明</h2>
    <p>依老師說明，整個「4. AI 協助要求」為選填，包含 GPT 輔助內容標示、AI 使用與對話紀錄、對話 PDF，以及 NotebookLM 研究與至少 100 字摘要。以上均不列入必做或待完成目標。</p>

    <h2>8. 重現與成果對照</h2>
    <pre>python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe scripts/analyze.py
.\\.venv\\Scripts\\python.exe scripts/build_report.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests
node tests/test_browser_core.cjs</pre>
    <p>此專案已附原始 CSV，可離線重跑分析；第一次建立環境仍需安裝套件。已建立的本機 <code>.venv</code> 可直接使用，不需重新安裝。版本：{html.escape(json.dumps(s['versions'],ensure_ascii=False))}。</p>
    {table(['題目要求','成果'],[
      ['Kaggle 14–39 個原始特徵','20 特徵 flood.csv；來源、固定鏡像與 SHA-256 已記錄'],
      ['CRISP-DM 六階段','本報告第 1–6 節與 Notebook'],
      ['五種特徵選擇、三種模型','18 組比較、selected_features.csv、validation_search.csv'],
      ['六項指標及五大假設','model_comparison.csv；全模型退化註記與 14 特徵補充診斷'],
      ['結果圖、95% 預測區間與新樣本','SVG 圖表、split conformal 區間、互動圖、predict.py 與 new_predictions.csv'],
      ['網頁、資料集切換、任意 CSV','web/dist/index.html；單一分析區載入、訓練、下載'],
      ['第 4 項 AI 協助要求（選填）','含 GPT 使用紀錄、對話 PDF 與 NotebookLM 研究摘要；均不列入待完成目標'],
      ['GitHub / web 繳交','本機網頁已完成；repository 與已發布網站的繳交連結尚待確認'],
      ['報告基本資料','姓名、學號待填，並同步更新 HTML／PDF']])}
    <h2>9. 參考來源</h2>
    <ol>{''.join('<li><a href="'+url+'">'+title+'</a></li>' for title,url in refs)}</ol>
    <p class="small">來源查閱：2026-10-22。分析與網頁均可離線使用；外部來源連結需網路。資料授權／來源以原提供者資訊為準，本專案未重新宣稱原始資料的授權。</p>
    '''
    report=document('AIoT HW2｜洪水機率多元線性迴歸研究報告',body)
    (REPORTS/'HW2_研究報告.html').write_text(report.replace('href="index.html"','href="../web/dist/index.html"'),encoding='utf-8')
    (WEB/'report.html').write_text(report,encoding='utf-8')
    shutil.copyfile(ROOT/'results/model_comparison.csv',WEB/'model_comparison.csv')
    (ROOT/'data/SOURCE.md').write_text('# 資料來源與版本\n\n'+json.dumps(s['source'],ensure_ascii=False,indent=2)+'\n\nSHA-256: '+s['sha256']+'\n\n50000 rows; 20 predictors; FloodProbability target. CSV line endings may differ from Git blob; local SHA-256 records the file used in this experiment.\n\n50_Startups.csv 來源： https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv\n原始 4 個輸入（含 State 類別）僅作匯入範例，不作主資料。\n',encoding='utf-8')
    build_notebook(s)
    transcript=ROOT/'reports/conversation_snapshot.json'
    if transcript.exists():
        entries=json.loads(transcript.read_text(encoding='utf-8'))
        log='<p class="eyebrow">AI ASSISTANCE DISCLOSURE</p><h1>AI 協助與對話紀錄</h1><div class="note">本檔是目前對話中可見文字的本機整理快照，包含需求、回覆與進度說明；不包含工具輸出、系統訊息、後續新增回合，不冒稱平台原生的完整匯出。原始題目截圖保留於「題目」資料夾。若需完整對話，請從本對話介面匯出或列印。</div><h2>GPT 輔助範圍</h2><p>讀題、資料來源、實驗流程、程式撰寫、執行分析、診斷解讀、網頁與報告。NotebookLM 尚未實際操作；研究準備與 GPT 摘要草稿另列。</p><h2>使用者可見對話快照</h2>'
        for e in entries:
            log+='<div class="entry"><p class="role">'+html.escape(e['role'])+'</p><div class="transcript">'+html.escape(e['text'])+'</div></div>'
        (REPORTS/'AI_協助與對話紀錄.html').write_text(document('AI 協助與對話紀錄',log),encoding='utf-8')
    print('Built report.html, provenance and HW2_Multiple_Linear_Regression.ipynb')


def build_notebook(s):
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
    def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True)))
    md('# HW2：洪水機率多元線性迴歸\n\n依 CRISP-DM 六階段進行。主資料為 50,000 列、20 個原始特徵，Notebook 呼叫可獨立執行的完整分析程式。請選擇專案的 `.venv` Python kernel，從頭依序執行。')
    code('from pathlib import Path\nimport json, sys\nimport pandas as pd\nROOT = Path.cwd()\nif not (ROOT / "scripts/analyze.py").exists():\n    raise RuntimeError("請把 Notebook 工作目錄設為專案根目錄")\nsys.path.insert(0, str(ROOT / "scripts"))\n')
    md('## 1. Business Understanding\n\n目標是連續 FloodProbability 分數，不二元化；建立可解釋的教學模型。資料沒有可驗證的地理時間來源，不能直接當作真實防災系統。')
    md('## 2. Data Understanding\n\nKaggle：'+s['source']['kaggle']+'\n\n實際使用固定 commit 的公開鏡像。來源與 SHA-256 詳見 data/SOURCE.md。')
    code('df = pd.read_csv(ROOT / "data/flood.csv")\nprint("shape:", df.shape, "missing:", df.isna().sum().sum(), "duplicates:", df.duplicated().sum())\nprint(df.describe().T)\n')
    md('## 3. Data Preparation\n\n先切分 60/20/20，seed=42。中位數補值、標準化、selector 只根據訓練集。驗證集用於挑 alpha 和模型，測試集只作最終評估。SFS 在訓練子樣本 5,000 列做三折 backward 選擇；fold 內 preprocessing。')
    md('## 4. Modeling\n\n以下實際執行五種特徵選擇與三種模型的完整實驗。程式碼可在 scripts/analyze.py 檢閱；不只讀取預先產生結果。')
    code('import analyze\nanalyze.main()\nsummary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))\n')
    code('print(pd.read_csv(ROOT / "results/selected_features.csv").to_string(index=False))\nprint(pd.read_csv(ROOT / "results/validation_search.csv").to_string(index=False))\n')
    md('## 5. Evaluation\n\n六項指標、五項假設與殘差圖。完整模型殘差近零，四項檢定退化；另呈現 14 特徵模型診斷。未拒絕假設不等於證明成立；無時間／群組資訊時不能確認獨立性。')
    code('print(pd.read_csv(ROOT / "results/model_comparison.csv").to_string(index=False))\nprint("Selected by validation:", summary["best"]["key"])\nprint("Mean-only baseline:", summary["baseline"])\nprint("Post-model descriptive identity check:", summary["deterministic_check"])\n')
    code('print(pd.DataFrame(summary["diagnostics"]).to_string(index=False))\nprint(pd.DataFrame(summary["diagnostics_reduced14"]).to_string(index=False))\n')
    md('### 95% 預測區間\n\n固定訓練資料決定的 SelectKBest14 / OLS，不用驗證目標選此輔助模型，使用 validation 作 split conformal calibration；test 僅評估涵蓋率，未調整寬度。完整模型的近零殘差區間退化，另以 14 特徵區間呈現不確定性。')
    code('print(summary["prediction_interval"])\nplots = ["actual_vs_predicted.svg", "residuals.svg", "prediction_intervals.svg", "residuals_reduced14.svg", "qq_plot.svg"]\ntry:\n    from IPython.display import display, SVG\n    for name in plots:\n        display(SVG(filename=str(ROOT / "results" / name)))\nexcept ImportError:\n    for name in plots:\n        print("SVG plot:", ROOT / "results" / name)\n')
    md('## 6. Deployment\n\n用同一組訓練參數預測新樣本；完整輸出不裁切到 [0,1]。互動頁使用單一分析區，下拉選單可切換 Flood、50 Startups 或自選 CSV。')
    code('from predict import predict_frame\nmodels = json.loads((ROOT / "results/models.json").read_text(encoding="utf-8"))\nnew = pd.read_csv(ROOT / "data/new_samples.csv")\nnew["prediction"] = predict_frame(new, models[summary["best"]["key"]])\nprint(new.to_string(index=False))\n')
    md('## 選填項目說明\n\n依老師說明，整個「4. AI 協助要求」為選填，包含 GPT 輔助內容標示、AI 使用與對話紀錄、對話 PDF，以及 NotebookLM 研究與至少 100 字摘要。以上均不列入必做或待完成目標。')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (HW2 .venv)',language='python',name='python3'),language_info=dict(name='python',version=s['versions']['python'])),nbformat=4,nbformat_minor=5)
    # Cell IDs are required for nbformat 4.5.
    for i,c in enumerate(cells):c['id']=f'hw2-cell-{i:02d}'
    (ROOT/'HW2_Multiple_Linear_Regression.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
