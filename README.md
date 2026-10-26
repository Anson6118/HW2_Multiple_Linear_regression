# AIoT HW2：Multiple Linear Regression

主資料：Kaggle Flood Prediction Dataset 的 `flood.csv` 公開鏡像，50,000 筆、**20 個原始輸入特徵**，目標 `FloodProbability`。涵蓋 CRISP-DM、五種特徵選擇、Linear Regression / Ridge / Lasso、六種評估指標、五項假設檢查、預測圖及新樣本預測。

## 直接查看

- **互動網頁**：[web/dist/index.html](web/dist/index.html)，可直接雙擊開啟，不需啟動伺服器。
- **研究報告**：[reports/HW2_研究報告.html](reports/HW2_研究報告.html)；對應 PDF 為 `reports/HW2_研究報告.pdf`。
- PowerShell 可執行 `./開啟網頁.ps1` 開啟本機網頁。

網站使用單一分析區，透過下拉選單選擇 **Flood、50 Startups 或自選資料集**。內建資料會自動載入與分析；選「自選資料集」才顯示 CSV 檔案／公開網址匯入欄位。可選擇數值目標、設定 Linear/Ridge/Lasso 和 alpha、選擇全部或前 14 個特徵、下載測試集預測。網址來源若限制跨來源存取（CORS），可先下載再匯入。50 Startups 僅示範 CSV 匯入，主作業特徵條件由洪水資料滿足。

## GitHub Pages 部署

網站：<https://anson6118.github.io/HW2_Multiple_Linear_regression/>

部署方式：**Settings → Pages → Build and deployment → Source** 設為 **Deploy from a branch**，Branch 選 **main / (root)**。根目錄的 `index.html` 會自動導向 `web/dist/`，`.nojekyll` 讓檔案原樣發布。推送至 `main` 後 GitHub 會自動重新部署。

`.github/workflows/validate.yml` 會在網頁檔案變更時執行功能檢查（不負責部署）。

## 研究結果與重要限制

固定 seed=42，60% 訓練、20% 驗證、20% 測試；選擇與補值／標準化只擬合訓練資料，驗證集決定 alpha 和勝出組合。**以驗證 RMSE 選出的 All20 / LinearRegression，測試 R²=1，RMSE 約 8.55×10⁻¹⁶。** 14 特徵線性版本的測試 R² 約 0.70。

本檔案的目標值幾乎精確等於 `0.005 × 20 個輸入的總和`，所以完美分數反映生成規則，不能作為真實防災能力的證據。完整模型殘差接近浮點誤差，四項假設檢定退化，另以 SelectKBest14 / OLS 訓練殘差補充診斷。未宣稱五項假設全數成立，也不以無時間意義的 DW 證明獨立性。

題目要求的**預測區間圖**已實作：固定 SelectKBest14 / OLS，用 10,000 筆 validation 校準 95% split conformal 區間，另外以 test 檢查涵蓋率，沒有用 test 調寬度。見 `results/prediction_intervals.svg / .csv`、報告與網頁。此區間屬 14 特徵模型，不與最佳 All20 模型的點預測混用；交換性假設與退化區間限制已說明。

實際資料來源是固定 GitHub commit 的鏡像，詳見 [data/SOURCE.md](data/SOURCE.md)；沒有宣稱以 Kaggle API 下載核對。原始題目三張截圖完整保留在 `題目/`。

## 重跑分析

目前工作區已建好 `.venv`，可直接執行：

```powershell
.\.venv\Scripts\python.exe scripts/analyze.py
.\.venv\Scripts\python.exe scripts/build_report.py
.\.venv\Scripts\python.exe scripts/predict.py data/new_samples.csv --output results/predictions.csv
```

其他電腦第一次設定：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

分析檔案已隨附，重跑不用下載資料。依需求可執行 `./重跑分析.ps1`。PDF 重建需要本機 Chromium 瀏覽器：

```powershell
.\.venv\Scripts\python.exe scripts/export_pdf.py
```

## 驗證

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node tests/test_browser_core.cjs
.\.venv\Scripts\python.exe scripts/browser_qa.py
```

Python 測試包含切分互斥、訓練參數一致性、模型選擇與報告指標、批次預測、JS/Python 迴歸數值一致性。瀏覽器 QA 操作單一分析區、資料集／模型／特徵切換、單筆預測、CSV 上傳、錯誤處理與下載。相關紀錄、桌面與手機截圖保存在 `results/`。

## 主要檔案

| 路徑 | 用途 |
|---|---|
| `scripts/analyze.py` | 完整分析：5 種 selector × 3 模型，加 All20 基準，共 18 組 |
| `scripts/predict.py` | 用已訓練的模型批次預測 CSV |
| `scripts/build_report.py` | 從實際結果建立中文報告、Notebook、資料來源文件 |
| `web/dist/` | 完整離線互動網頁；不需要 CDN 或額外套件 |
| `results/model_comparison.csv` | 最終測試指標及驗證 RMSE |
| `results/validation_search.csv` | 所有候選 alpha 的驗證成績 |
| `results/split_manifest.csv` | 每筆原始資料的 train/validation/test 分割 |
| `results/selected_features.csv` | 各 selector 入選欄位 |
| `results/models.json` | 18 組模型的訓練中位數、平均值、標準差、係數及截距 |
| `results/test_predictions.csv` | 每筆測試資料的真值及 18 組模型預測 |
| `results/assumptions*.json` | 全特徵退化診斷與 14 特徵補充診斷 |
| `results/*.svg` | 可獨立使用的預測與殘差圖 |
| `results/prediction_interval_summary.json` | 95% 區間校準設定、半寬與獨立測試涵蓋率 |

## 特徵選擇設定

- SelectKBest：`f_regression` 前 14。
- RFE：以 Ridge(alpha=1) 遞迴淘汰至 14。
- Backward SFS：固定抽樣 5,000 筆**訓練資料**，三折 CV，每折各自補值與標準化，選 14。
- Lasso：標準化後 alpha=.001，按絕對係數取前 14；不是聲稱只剩 14 個非零係數。
- Permutation：外層訓練集內部 75% 擬合／25% 排名，5 次打亂，按 RMSE 變化取前 14。

只有 20 個原始特徵，比較 14 與 20，不造出 25/30 特徵湊數。Ridge alpha∈{.01,.1,1,10,100}；Lasso alpha∈{1e-6,1e-5,1e-4,.001,.01}。不在 train+validation 重訓，確保匯出的模型與測試報告對應。
