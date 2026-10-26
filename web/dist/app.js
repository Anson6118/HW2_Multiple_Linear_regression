/* Browser-local regression lab. No data is sent to a server. */
(() => {
  'use strict';
  const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
  const median = a => {const b=[...a].sort((x,y)=>x-y), n=b.length; return n ? (n%2 ? b[(n-1)/2] : (b[n/2-1]+b[n/2])/2) : 0;};
  const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = v => v == null || !Number.isFinite(v) ? '—' : (v!==0 && Math.abs(v)<.0001 ? v.toExponential(2) : Math.abs(v)>=1000 ? v.toLocaleString('en-US',{maximumFractionDigits:2}) : v.toFixed(5));
  function parseCSV(text) {
    text=String(text).replace(/^\uFEFF/,'');
    const rows=[]; let row=[], value='', quoted=false;
    for(let i=0;i<text.length;i++) {
      const c=text[i];
      if(c==='"') {
        if(quoted && text[i+1]==='"') {value+='"'; i++;}
        else if(quoted || value==='') quoted=!quoted;
        else throw Error('CSV 引號位置不正確。');
      } else if(c===',' && !quoted) {row.push(value); value='';}
      else if((c==='\n'||c==='\r') && !quoted) {
        if(c==='\r'&&text[i+1]==='\n') i++;
        row.push(value); if(row.some(v=>v.trim()!=='')) rows.push(row); row=[]; value='';
      } else value+=c;
    }
    if(quoted) throw Error('CSV 有未關閉的引號。');
    row.push(value); if(row.some(v=>v.trim()!=='')) rows.push(row);
    if(rows.length<2) throw Error('CSV 至少需要標題列和一筆資料。');
    const headers=rows.shift().map(v=>v.trim());
    if(headers.some(v=>!v) || new Set(headers).size!==headers.length) throw Error('欄名不能空白或重複。');
    if(rows.some(r=>r.length!==headers.length)) throw Error('CSV 每列欄位數必須與標題一致。');
    return {headers,rows};
  }
  function shuffleIndices(n,seed=42) {
    let state=seed>>>0;
    const random=()=>{state=(1664525*state+1013904223)>>>0;return state/4294967296;};
    const a=Array.from({length:n},(_,i)=>i);
    for(let i=n-1;i>0;i--){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}
    return a;
  }
  // Partial-pivot Gaussian elimination; all callers use standardized predictors.
  function solve(A,b) {
    const n=b.length, m=A.map((r,i)=>[...r,b[i]]);
    for(let j=0;j<n;j++) {
      let pivot=j;
      for(let k=j+1;k<n;k++) if(Math.abs(m[k][j])>Math.abs(m[pivot][j])) pivot=k;
      if(Math.abs(m[pivot][j])<1e-12) throw Error('特徵矩陣秩不足，請改用 Ridge 或移除共線欄位。');
      [m[j],m[pivot]]=[m[pivot],m[j]];
      const d=m[j][j]; for(let k=j;k<=n;k++) m[j][k]/=d;
      for(let i=j+1;i<n;i++) {const c=m[i][j];for(let k=j;k<=n;k++) m[i][k]-=c*m[j][k];}
    }
    const x=Array(n).fill(0);
    for(let i=n-1;i>=0;i--){x[i]=m[i][n];for(let j=i+1;j<n;j++)x[i]-=m[i][j]*x[j];}
    return x;
  }
  function fitModel(X,y,type='LinearRegression',alpha=1) {
    const n=X.length,p=X[0].length,intercept=mean(y);
    const yc=y.map(v=>v-intercept), gram=Array.from({length:p},()=>Array(p).fill(0)),xy=Array(p).fill(0);
    for(let i=0;i<n;i++) for(let j=0;j<p;j++) {xy[j]+=X[i][j]*yc[i];for(let k=0;k<=j;k++)gram[j][k]+=X[i][j]*X[i][k];}
    for(let j=0;j<p;j++)for(let k=0;k<j;k++)gram[k][j]=gram[j][k];
    let coef;
    if(type==='Lasso') {
      // Same objective as sklearn: (1/2n)||y-Xw||² + alpha ||w||_1.
      coef=Array(p).fill(0); let converged=false;
      for(let it=0;it<10000;it++) {
        let maxDelta=0;
        for(let j=0;j<p;j++) {
          let rho=xy[j];for(let k=0;k<p;k++)if(k!==j)rho-=gram[j][k]*coef[k];
          const next=Math.sign(rho)*Math.max(0,Math.abs(rho)-n*alpha)/gram[j][j];
          maxDelta=Math.max(maxDelta,Math.abs(next-coef[j]));coef[j]=next;
        }
        if(maxDelta<1e-9*Math.max(1,...coef.map(Math.abs))) {converged=true;break;}
      }
      if(!converged) throw Error('Lasso 未收斂；請增加 alpha 或簡化資料。');
    } else {
      // Ridge objective: ||y-Xw||² + alpha ||w||²; intercept not penalized.
      const matrix=gram.map((r,j)=>r.map((v,k)=>v+(j===k && type==='Ridge'?alpha:0)));
      coef=solve(matrix,xy);
    }
    return {coef,intercept,predict:row=>dot(row,coef)+intercept};
  }
  function score(y,pred,p) {
    const n=y.length,ym=mean(y),mse=mean(y.map((v,i)=>(v-pred[i])**2)),sst=y.reduce((s,v)=>s+(v-ym)**2,0);
    const r2=sst>0 ? 1-n*mse/sst : null, nz=y.map((v,i)=>[v,i]).filter(([v])=>Math.abs(v)>1e-12);
    return {MAE:mean(y.map((v,i)=>Math.abs(v-pred[i]))),MSE:mse,RMSE:Math.sqrt(mse),R2:r2,
      Adjusted_R2:n>p+1&&r2!=null ? 1-(1-r2)*(n-1)/(n-p-1):null,
      MAPE:nz.length?mean(nz.map(([v,i])=>Math.abs((v-pred[i])/v)))*100:null,MAPE_excluded_zeros:n-nz.length};
  }
  function trainCSV(dataset,target,modelType='Ridge',selection='all',alpha=1) {
    if(!Number.isFinite(alpha)||alpha<0||(modelType==='Lasso'&&alpha===0)) throw Error('alpha 必須非負；Lasso 請使用大於 0 的值。');
    const targetIndex=dataset.headers.indexOf(target);
    if(targetIndex<0) throw Error('請選擇目標欄位。');
    let omitted=0;
    const source=dataset.rows.map((row,i)=>({row,id:i+2})).filter(({row})=>{
      const v=row[targetIndex].trim(),ok=v!==''&&Number.isFinite(Number(v));if(!ok)omitted++;return ok;
    });
    if(source.length<20) throw Error('至少需要 20 筆具有數值目標的資料。');
    if(source.length>50000) throw Error('瀏覽器實驗最多 50,000 筆，完整大資料請使用 Python 分析。');
    const ids=shuffleIndices(source.length),a=Math.floor(ids.length*.6),b=Math.floor(ids.length*.8);
    const train=ids.slice(0,a).map(i=>source[i]),val=ids.slice(a,b).map(i=>source[i]),test=ids.slice(b).map(i=>source[i]);
    const rawIndices=dataset.headers.map((h,i)=>i).filter(i=>i!==targetIndex&&!/^(id|index|row_id)$/i.test(dataset.headers[i]));
    if(rawIndices.length>39) throw Error('最多支援 39 個原始輸入欄位；請先移除不需要的欄位。');
    const schema=[];
    for(const i of rawIndices) {
      const values=train.map(x=>x.row[i].trim()).filter(x=>x!==''&&x!=='?');
      if(!values.length) continue;
      if(values.every(v=>Number.isFinite(Number(v)))) schema.push({i,name:dataset.headers[i],numeric:true,median:median(values.map(Number))});
      else {
        const levels=[...new Set(values)].sort();
        if(levels.length>20) throw Error(`「${dataset.headers[i]}」有超過 20 個類別，請移除文字識別欄。`);
        schema.push({i,name:dataset.headers[i],numeric:false,levels});
      }
    }
    const encodedNames=schema.flatMap(c=>c.numeric?[c.name]:c.levels.slice(1).map(v=>`${c.name}=${v}`));
    if(!encodedNames.length||encodedNames.length>100)throw Error('編碼後需有 1–100 個特徵。');
    const encode=({row})=>schema.flatMap(c=>{
      const v=row[c.i].trim();
      return c.numeric?[v===''||v==='?'?c.median:Number(v)]:c.levels.slice(1).map(level=>v===level?1:0);
    });
    const rawX=train.map(encode), ytrain=train.map(x=>Number(x.row[targetIndex]));
    if(rawX.flat().some(x=>!Number.isFinite(x))) throw Error('數值特徵的驗證／測試資料含非數值內容，請清理 CSV。');
    const means=encodedNames.map((_,j)=>mean(rawX.map(r=>r[j]))), scales=encodedNames.map((_,j)=>Math.sqrt(mean(rawX.map(r=>(r[j]-means[j])**2))));
    const usable=encodedNames.map((_,j)=>j).filter(j=>scales[j]>1e-12);
    if(!usable.length)throw Error('所有訓練特徵均為常數，無法建模。');
    const ym=mean(ytrain),yvar=ytrain.reduce((s,v)=>s+(v-ym)**2,0);
    const rank=usable.map(j=>({j,importance:Math.abs(rawX.reduce((s,r,i)=>s+(r[j]-means[j])*(ytrain[i]-ym),0)/Math.sqrt(rawX.length*scales[j]**2*yvar))})).sort((a,b)=>(b.importance-a.importance)||(a.j-b.j));
    const selected=(selection==='top14'?rank.slice(0,14).map(v=>v.j):usable).sort((a,b)=>a-b);
    if(selected.length>=train.length-1)throw Error('樣本相對特徵太少，請增加資料或選擇較少特徵。');
    const transform=x=>{const row=encode(x);const z=selected.map(j=>(row[j]-means[j])/scales[j]);if(z.some(v=>!Number.isFinite(v)))throw Error('數值欄位在驗證／測試集出現非數值。');return z;};
    const fitted=fitModel(train.map(transform),ytrain,modelType,alpha);
    const yval=val.map(x=>Number(x.row[targetIndex])),ypval=val.map(x=>fitted.predict(transform(x)));
    const yt=test.map(x=>Number(x.row[targetIndex])),yp=test.map(x=>fitted.predict(transform(x)));
    const features=selected.map(j=>encodedNames[j]);
    const rows=test.map((r,i)=>({csv_row:r.id,actual:yt[i],predicted:yp[i],residual:yt[i]-yp[i]}));
    return {metrics:score(yt,yp,features.length),validation:score(yval,ypval,features.length),features,rows,
      split:{train:train.length,validation:val.length,test:test.length},omitted,rawCount:rawIndices.length,
      droppedConstants:encodedNames.length-usable.length,schema,means,scales,selected,
      coefficients:fitted.coef,intercept:fitted.intercept,modelType,alpha,
      note:'類別以訓練集建立 one-hot 並省略首類；未知／缺失類別使用基準編碼。補值、標準化、排名均只使用訓練資料。'};
  }
  globalThis.HW2Core={parseCSV,shuffleIndices,solve,fitModel,score,trainCSV};
  if(typeof document==='undefined')return;
  const D=window.HW2_RESULTS;
  if(!D){document.body.textContent='缺少 results.js，請先執行 scripts/analyze.py。';return;}
  const $=s=>document.querySelector(s);
  function chart(points,{residual=false,compact=false}={}) {
    if(!points.length)return '<div class="empty-chart">執行分析後顯示圖表</div>';
    const x=points.map(p=>residual?p.predicted:p.actual),y=points.map(p=>residual?p.actual-p.predicted:p.predicted);
    let xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...y),ymax=Math.max(...y);
    if(!residual){xmin=ymin=Math.min(xmin,ymin);xmax=ymax=Math.max(xmax,ymax);}
    const dx=(xmax-xmin)*.08||.01,dy=(ymax-ymin)*.08||1e-15;
    xmin-=dx;xmax+=dx;ymin-=dy;ymax+=dy;
    const W=600,H=compact?320:380,L=66,R=22,T=23,B=55;
    const px=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),py=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
    let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${residual?'殘差圖':'實際與預測散佈圖'}"><rect width="100%" height="100%" fill="white"/>`;
    for(let i=0;i<=4;i++) {
      const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;
      s+=`<path d="M${L} ${py(yv)}H${W-R} M${px(xv)} ${T}V${H-B}" stroke="#e5eeea"/><text x="${px(xv)}" y="${H-B+20}" text-anchor="middle" fill="#70817c" font-family="Arial" font-size="10">${axis(xv)}</text><text x="${L-9}" y="${py(yv)+3}" text-anchor="end" fill="#70817c" font-family="Arial" font-size="10">${axis(yv)}</text>`;
    }
    const stride=Math.max(1,Math.ceil(points.length/1200));
    for(let i=0;i<x.length;i+=stride)s+=`<circle cx="${px(x[i]).toFixed(2)}" cy="${py(y[i]).toFixed(2)}" r="2.1" fill="#008a88" opacity=".38"/>`;
    s+=residual&&ymin<=0&&ymax>=0?`<path d="M${L} ${py(0)}H${W-R}" stroke="#d68b53" stroke-dasharray="5 5"/>`:!residual?`<path d="M${px(xmin)} ${py(xmin)}L${px(xmax)} ${py(xmax)}" stroke="#d68b53" stroke-dasharray="5 5"/>`:'';
    return s+`<text x="${W/2}" y="${H-10}" text-anchor="middle" fill="#70817c" font-family="Arial" font-size="11">${residual?'Predicted value':'Actual value'}</text><text transform="translate(16 ${H/2}) rotate(-90)" text-anchor="middle" fill="#70817c" font-family="Arial" font-size="11">${residual?'Residual (actual - predicted)':'Predicted value'}</text></svg>`;
  }
  function axis(v){return Math.abs(v)>=10000?(v/1000).toFixed(0)+'k':v!==0&&Math.abs(v)<.0001?v.toExponential(1):v.toFixed(2);}
  const modelPredict=(x,m)=>m.intercept+m.features.reduce((sum,f,i)=>sum+((x[f]??m.median[i])-m.mean[i])/m.scale[i]*m.coefficients[i],0);
  const metricCards=m=>['MAE','MSE','RMSE','R2','Adjusted_R2','MAPE'].map(key=>`<div class="metric"><span>${key==='R2'?'R²':key==='Adjusted_R2'?'ADJUSTED R²':key==='MAPE'?'MAPE (%)':key}</span><strong>${fmt(m[key])}</strong></div>`).join('');
  $('#hero-chart').innerHTML=chart(D.scatter,{compact:true});$('#hero-r2').textContent=D.summary.best.R2.toFixed(5);
  $('#model-select').innerHTML=D.summary.models.map(r=>`<option value="${esc(r.key)}">${esc(r.key)}${r.key===D.summary.best.key?' · 驗證集最佳':''}</option>`).join('');
  $('#model-select').value=D.summary.best.key;
  let activeChart='prediction';
  function renderModel(){
    const key=$('#model-select').value,row=D.summary.models.find(r=>r.key===key),model=D.models[key];
    $('#metric-cards').innerHTML=metricCards(row);
    $('#feature-list').innerHTML=model.features.map(f=>`<span class="feature-chip">${esc(f)}</span>`).join('');
    const points=D.test_sample.map(r=>({actual:r.actual,predicted:modelPredict(r.features,model)}));
    $('#model-chart').innerHTML=chart(points,{residual:activeChart==='residual'});
  }
  $('#model-select').addEventListener('change',renderModel);
  document.querySelectorAll('[data-chart]').forEach(b=>b.addEventListener('click',()=>{activeChart=b.dataset.chart;document.querySelectorAll('[data-chart]').forEach(t=>t.setAttribute('aria-pressed',t===b));renderModel();}));
  renderModel();
  $('#comparison-table').innerHTML='<thead><tr><th>特徵選擇 / 模型</th><th>p</th><th>alpha</th><th>Val RMSE</th><th>Test MAE</th><th>Test MSE</th><th>Test RMSE</th><th>Test R²</th><th>Adjusted R²</th><th>MAPE %</th></tr></thead><tbody>'+D.summary.models.map(r=>`<tr class="${r.key===D.summary.best.key?'winner':''}"><td>${esc(r.key)}</td><td>${r.features}</td><td>${r.alpha??'—'}</td>${['validation_RMSE','MAE','MSE','RMSE','R2','Adjusted_R2','MAPE'].map(k=>`<td>${fmt(r[k])}</td>`).join('')}</tr>`).join('')+'</tbody>';
  $('#assumptions-table').innerHTML='<thead><tr><th>假設</th><th>檢查方式</th><th>統計量</th><th>p-value</th><th>判讀</th></tr></thead><tbody>'+D.summary.diagnostics_reduced14.map(r=>`<tr><td>${esc(r.assumption)}</td><td>${esc(r.test)}</td><td>${fmt(r.statistic)}</td><td>${fmt(r.p_value)}</td><td>${esc(r.conclusion)}</td></tr>`).join('')+'</tbody>';
  const bestModel=D.models[D.summary.best.key];
  const interval=D.summary.prediction_interval;
  $('#interval-stats').innerHTML=`<div class="metric"><span>名目預測區間</span><strong>95%</strong></div><div class="metric"><span>TEST 實際涵蓋率</span><strong>${(100*interval.test_coverage).toFixed(2)}%</strong></div><div class="metric"><span>平均區間寬度</span><strong>${fmt(interval.mean_width)}</strong></div>`;
  $('#prediction-form').innerHTML=D.summary.features.map((f,i)=>{const m=bestModel.features.indexOf(f),v=m>=0?bestModel.median[m]:5;return `<label>${esc(f)}<input name="${esc(f)}" data-feature="${esc(f)}" type="number" step="any" value="${v}" required aria-label="${esc(f)}"></label>`;}).join('');
  function updatePrediction(){
    const x={};let invalid=false,outside=false;
    document.querySelectorAll('[data-feature]').forEach(el=>{const f=el.dataset.feature,v=el.valueAsNumber;invalid||=!Number.isFinite(v);x[f]=v;outside||=v<D.training_ranges[f][0]||v>D.training_ranges[f][1];});
    if(invalid){$('#prediction-value').textContent='—';$('#prediction-interval').textContent='';$('#prediction-note').textContent='請為每個因子填入有效數值。';return;}
    const pred=modelPredict(x,bestModel);$('#prediction-value').textContent=pred.toFixed(4);
    const reduced=modelPredict(x,D.models[interval.model]);
    $('#prediction-interval').textContent=`另以 14 特徵 OLS 預測：${reduced.toFixed(4)}；95% 預測區間 [${(reduced-interval.radius).toFixed(4)}, ${(reduced+interval.radius).toFixed(4)}]。`;
    $('#prediction-note').textContent=outside?'部分輸入超出訓練資料範圍，屬外插預測。':pred<0||pred>1?'線性模型輸出超出 [0,1]；此處保留原始預測，未裁切。':'使用 All20 / LinearRegression；顯示原始預測，未裁切。';
  }
  $('#prediction-form').addEventListener('input',updatePrediction);$('#prediction-form').addEventListener('submit',e=>e.preventDefault());
  $('#reset-inputs').addEventListener('click',()=>{$('#prediction-form').reset();updatePrediction();});updatePrediction();
  function makePanel(example){
    const panel=document.createElement('article');panel.className='card lab-panel';panel.id='analysis-lab';
    panel.innerHTML=`<div class="panel-head"><h3>資料分析</h3><span>REGRESSION LAB</span></div><label class="dataset-picker">選擇資料集<select data-dataset aria-label="資料集"><option value="flood">Flood</option><option value="startups">50 Startups</option><option value="custom">自選資料集</option></select></label><div class="custom-dataset" hidden><label class="file-drop">匯入 CSV（UTF-8，最多 10 MB）<input type="file" accept=".csv,text/csv" aria-label="匯入 CSV"></label><div class="csv-url"><label>或貼上 CSV 網址<input type="url" placeholder="https://…/dataset.csv" aria-label="CSV 網址"></label><button type="button" class="button secondary" data-load-url>從網址載入</button></div></div><div class="lab-controls"><label>預測目標<select data-control="target" aria-label="預測目標"></select></label><label>模型<select data-control="model" aria-label="模型"><option>LinearRegression</option><option selected>Ridge</option><option>Lasso</option></select></label><label>特徵選擇<select data-control="selection" aria-label="特徵選擇"><option value="all">全部（移除常數）</option><option value="top14">相關性前 14（最多）</option></select></label><label>alpha<input type="number" data-control="alpha" aria-label="alpha" min="0" step="any" value="1"></label></div><div class="lab-tools"><button class="button" data-run>訓練並評估 →</button><button class="button secondary" data-download disabled>下載預測 CSV ↓</button></div><p class="lab-status" role="status" aria-live="polite"></p><div class="lab-result"></div>`;
    const setup=document.createElement('div');setup.className='lab-setup';
    Array.from(panel.children).filter(child=>!child.matches('.panel-head,.lab-result')).forEach(child=>setup.append(child));
    panel.querySelector('.lab-result').before(setup);
    $('#lab-panels').append(panel);
    const el=s=>panel.querySelector(s),controls=name=>el(`[data-control="${name}"]`);
    let dataset,result,fileName,loadVersion=0,runVersion=0,urlAbort;
    const status=(text,error=false)=>{el('.lab-status').textContent=text;el('.lab-status').classList.toggle('error',error);};
    function clearResult(){result=null;el('.lab-result').innerHTML='';el('[data-download]').disabled=true;}
    function resetDataset(){
      ++loadVersion;++runVersion;urlAbort?.abort();urlAbort=null;
      dataset=null;clearResult();controls('target').innerHTML='';controls('target').disabled=true;
      el('[data-run]').disabled=true;el('[data-load-url]').disabled=false;
      return loadVersion;
    }
    function load(text,name,target){
      dataset=parseCSV(text);fileName=name;clearResult();
      controls('target').innerHTML=dataset.headers.map(h=>`<option>${esc(h)}</option>`).join('');
      controls('target').value=target||dataset.headers.at(-1);
      controls('target').disabled=false;el('[data-run]').disabled=false;
      status(`${fileName} · ${dataset.rows.length.toLocaleString()} 列 / ${dataset.headers.length} 欄，已就緒。`);
    }
    function chooseDataset(){
      resetDataset();
      const kind=el('[data-dataset]').value,custom=kind==='custom';
      el('.custom-dataset').hidden=!custom;
      el('.panel-head h3').textContent=custom?'自選資料集':kind==='flood'?'洪水機率':'新創公司利潤';
      el('input[type=file]').value='';el('input[type=url]').value='';
      if(custom){status('請匯入 CSV 檔案，或貼上公開 CSV 網址。');return;}
      load(window.HW2_EXAMPLES[kind],kind==='flood'?'flood_sample.csv':'50_Startups.csv',kind==='flood'?'FloodProbability':'Profit');
      controls('alpha').value=kind==='flood'?'0.01':'1';run();
    }
    el('[data-dataset]').addEventListener('change',chooseDataset);
    el('input[type=file]').addEventListener('change',async e=>{
      const f=e.target.files[0];if(!f)return;
      const version=resetDataset();status('正在讀取 CSV 檔案…');
      try{
        if(f.size>10*1024*1024)throw Error('檔案超過 10 MB，請使用 Python 分析。');
        const text=await f.text();if(version!==loadVersion)return;load(text,f.name);
      }catch(err){if(version===loadVersion)status(err.message,true);}
    });
    el('[data-load-url]').addEventListener('click',async()=>{
      const button=el('[data-load-url]'),version=resetDataset();let timer;
      try{
        const url=new URL(el('input[type=url]').value);
        if(!['http:','https:'].includes(url.protocol))throw Error('請輸入 http 或 https CSV 網址。');
        if(url.username||url.password)throw Error('請使用不含帳號密碼的公開 CSV 網址。');
        button.disabled=true;status('正在讀取 CSV 網址…');
        const controller=new AbortController();urlAbort=controller;timer=setTimeout(()=>controller.abort(),15000);
        const response=await fetch(url.href,{signal:controller.signal,credentials:'omit'});
        if(!response.ok)throw Error(`伺服器回應 HTTP ${response.status}。`);
        if(Number(response.headers.get('content-length'))>10*1024*1024)throw Error('檔案超過 10 MB。');
        const text=await response.text();
        if(version!==loadVersion)return;
        if(new Blob([text]).size>10*1024*1024)throw Error('檔案超過 10 MB。');
        load(text,url.pathname.split('/').pop()||'remote.csv');
      }catch(err){if(version===loadVersion)status(`${err.message} 網址讀取失敗時，可先下載 CSV 再匯入檔案。`,true);}
      finally{clearTimeout(timer);if(version===loadVersion){button.disabled=false;urlAbort=null;}}
    });
    panel.querySelectorAll('[data-control]').forEach(c=>c.addEventListener('change',()=>{++runVersion;clearResult();el('[data-run]').disabled=!dataset;status(dataset?'設定已變更，請重新訓練。':'請先匯入自選資料集。');}));
    function renderResult(){
      el('.lab-result').innerHTML=`<div class="metric-grid">${metricCards(result.metrics)}</div>${chart(result.rows,{compact:true})}<p class="fine">Train ${result.split.train} / Validation ${result.split.validation} / Test ${result.split.test} · seed=42 · Val RMSE ${fmt(result.validation.RMSE)}</p><p class="fine">原始輸入 ${result.rawCount} 欄，編碼／選擇後 ${result.features.length} 欄。${result.omitted?`移除 ${result.omitted} 列無效目標。`:''}MAPE 排除 ${result.metrics.MAPE_excluded_zeros} 列零目標。</p><p class="fine">${result.rawCount<14?'本資料僅作 CSV 匯入示範，原始特徵數不符主作業的 14–39 欄要求。':''}此實驗 alpha 由使用者指定，未自動調參；模型只擬合 train。${esc(result.note)}</p><details><summary>查看使用特徵</summary><p class="fine">${result.features.map(esc).join(' · ')}</p></details>`;
    }
    async function run(){
      if(!dataset){status('請先載入 CSV。',true);return;}
      const runButton=el('[data-run]'),version=++runVersion;runButton.disabled=true;clearResult();status('正在訓練與評估…');
      await new Promise(resolve=>setTimeout(resolve,30));
      if(version!==runVersion)return;
      try{result=trainCSV(dataset,controls('target').value,controls('model').value,controls('selection').value,Number(controls('alpha').value));renderResult();el('[data-download]').disabled=false;status(`${fileName} · 完成，測試資料僅用於最後評估。`);}catch(err){status(err.message,true);}finally{if(version===runVersion)runButton.disabled=!dataset;}
    }
    el('[data-run]').addEventListener('click',run);
    el('[data-download]').addEventListener('click',()=>{
      if(!result)return;
      const csv='csv_row,actual,predicted,residual\r\n'+result.rows.map(r=>[r.csv_row,r.actual,r.predicted,r.residual].join(',')).join('\r\n');
      const url=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
    el('[data-dataset]').value=example;chooseDataset();
  }
  makePanel('flood');
})();
