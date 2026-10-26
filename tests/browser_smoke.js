/* Runs against the real DOM, same scripts and controls as the delivered page. */
(async()=>{
  const checks=[];
  const pause=ms=>new Promise(r=>setTimeout(r,ms));
  function assert(name,value){if(!value)throw Error(name);checks.push(name);}
  const $=s=>document.querySelector(s);
  const report=document.createElement('pre');report.id='browser-qa-result';report.style.cssText='white-space:pre-wrap;padding:20px;background:#eef5ee;';document.body.append(report);
  try{
    await pause(200);
    assert('Main page loads 18 models',$('#model-select').options.length===18);
    assert('Exactly one analysis area is rendered',document.querySelectorAll('.lab-panel').length===1&&document.querySelectorAll('[data-dataset]').length===1);
    assert('The default Flood dataset finishes training',$('#analysis-lab .lab-status').textContent.includes('完成'));
    const datasetSelect=$('#analysis-lab [data-dataset]');
    assert('Dataset selector offers Flood, 50 Startups and custom datasets',Array.from(document.querySelectorAll('[data-dataset]')).every(s=>Array.from(s.options).map(o=>o.textContent).join('|')==='Flood|50 Startups|自選資料集'));
    assert('Built-in datasets hide custom import fields',Array.from(document.querySelectorAll('.custom-dataset')).every(p=>p.hidden));
    datasetSelect.value='startups';datasetSelect.dispatchEvent(new Event('change'));await pause(100);
    assert('Selecting 50 Startups loads Profit and trains automatically',$('#analysis-lab [data-control=target]').value==='Profit'&&$('#analysis-lab .lab-status').textContent.includes('完成'));
    datasetSelect.value='flood';datasetSelect.dispatchEvent(new Event('change'));await pause(100);
    assert('Selecting Flood restores FloodProbability and its heading',$('#analysis-lab [data-control=target]').value==='FloodProbability'&&$('#analysis-lab .panel-head h3').textContent.includes('洪水機率'));
    const first=$('[data-feature="MonsoonIntensity"]'),initial=Number($('#prediction-value').textContent);
    first.value=Number(first.value)+1;first.dispatchEvent(new Event('input',{bubbles:true}));
    assert('Prediction input updates fitted-model output',Math.abs(Number($('#prediction-value').textContent)-initial-.005)<1e-7);
    $('#reset-inputs').click();
    assert('Reset restores training median prediction',Math.abs(Number($('#prediction-value').textContent)-initial)<1e-7);
    $('#model-select').value='SelectKBest14 / LinearRegression';$('#model-select').dispatchEvent(new Event('change'));
    assert('Model selector updates selected features',$('#feature-list').querySelectorAll('.feature-chip').length===14);
    $('[data-chart="residual"]').click();
    assert('Residual tab updates accessible chart',$('#model-chart svg').getAttribute('aria-label')==='殘差圖');
    datasetSelect.value='custom';datasetSelect.dispatchEvent(new Event('change'));
    assert('Custom selection reveals imports and clears old results',!$('#analysis-lab .custom-dataset').hidden&&!$('#analysis-lab .lab-result').textContent&&$('#analysis-lab [data-run]').disabled&&$('#analysis-lab [data-download]').disabled&&$('#analysis-lab [data-control=target]').options.length===0);
    // Real file-input import, not an internal function call.
    const dt=new DataTransfer();dt.items.add(new File([window.HW2_EXAMPLES.startups],'uploaded_startups.csv',{type:'text/csv'}));
    const file=$('#analysis-lab input[type=file]');file.files=dt.files;file.dispatchEvent(new Event('change',{bubbles:true}));
    await pause(100);
    assert('CSV upload recognizes Profit target',$('#analysis-lab [data-control=target]').value==='Profit');
    $('#analysis-lab [data-control=model]').value='LinearRegression';$('#analysis-lab [data-run]').click();await pause(150);
    assert('Uploaded CSV runs and produces chart',$('#analysis-lab .lab-status').textContent.includes('完成')&&!!$('#analysis-lab .lab-result svg'));
    $('#analysis-lab [data-control=model]').value='Lasso';$('#analysis-lab [data-control=alpha]').value='0';$('#analysis-lab [data-run]').click();await pause(100);
    assert('Invalid Lasso alpha reports an error',$('#analysis-lab .lab-status').classList.contains('error'));
    assert('Failed training disables stale download',$('#analysis-lab [data-download]').disabled);
    $('#analysis-lab [data-control=alpha]').value='1';$('#analysis-lab [data-run]').click();await pause(150);
    assert('Lasso retrains after correcting alpha',$('#analysis-lab .lab-status').textContent.includes('完成'));
    let clicked=false,blob;
    const oldCreate=URL.createObjectURL,oldClick=HTMLAnchorElement.prototype.click;
    URL.createObjectURL=b=>{blob=b;return oldCreate.call(URL,b);};HTMLAnchorElement.prototype.click=function(){clicked=true;};
    $('#analysis-lab [data-download]').click();await pause(30);
    URL.createObjectURL=oldCreate;HTMLAnchorElement.prototype.click=oldClick;
    assert('Download button creates prediction CSV',clicked&&blob instanceof Blob&&(await blob.text()).includes('actual,predicted,residual'));
    const bad=new DataTransfer();bad.items.add(new File(['x,x\n1,2'],'bad.csv',{type:'text/csv'}));file.files=bad.files;file.dispatchEvent(new Event('change',{bubbles:true}));await pause(100);
    assert('Malformed CSV is rejected with a visible message',$('#analysis-lab .lab-status').classList.contains('error'));
    assert('No horizontal overflow at current viewport',document.documentElement.scrollWidth<=window.innerWidth+1);
    const oldFetch=window.fetch;
    window.fetch=async()=>new Response(window.HW2_EXAMPLES.flood,{status:200,headers:{'content-type':'text/csv'}});
    $('#analysis-lab input[type=url]').value='https://example.test/flood.csv';$('#analysis-lab [data-load-url]').click();await pause(100);
    window.fetch=oldFetch;
    assert('CSV URL control loads response and recognizes target',$('#analysis-lab [data-control=target]').value==='FloodProbability');
    $('#analysis-lab input[type=url]').value='file:///private.csv';$('#analysis-lab [data-load-url]').click();await pause(50);
    assert('CSV URL rejects unsupported protocols',$('#analysis-lab .lab-status').classList.contains('error'));
    let release;
    window.fetch=()=>new Promise(resolve=>{release=()=>resolve(new Response(window.HW2_EXAMPLES.flood,{status:200}));});
    $('#analysis-lab input[type=url]').value='https://example.test/delayed.csv';$('#analysis-lab [data-load-url]').click();
    datasetSelect.value='startups';datasetSelect.dispatchEvent(new Event('change'));release();await pause(100);window.fetch=oldFetch;
    assert('Switching dataset ignores a previous pending URL response',$('#analysis-lab [data-control=target]').value==='Profit'&&$('#analysis-lab .custom-dataset').hidden&&$('#analysis-lab .lab-status').textContent.includes('50_Startups.csv'));
    assert('Returning from custom data trains the built-in dataset',$('#analysis-lab .lab-status').textContent.includes('完成')&&!!$('#analysis-lab .lab-result svg'));
    assert('Dataset selector and import fields fit the viewport',document.documentElement.scrollWidth<=window.innerWidth+1);
    report.textContent=JSON.stringify({status:'PASS',checks:checks.length,viewport:[innerWidth,innerHeight],names:checks},null,2);document.title='BROWSER_QA_PASS';
  }catch(error){report.textContent=JSON.stringify({status:'FAIL',checks,error:error.message},null,2);document.title='BROWSER_QA_FAIL';}
})();
