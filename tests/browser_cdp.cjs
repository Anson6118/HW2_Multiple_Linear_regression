/* Actual browser integration using Chromium's local DevTools protocol. */
'use strict';
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {spawn}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const profile=path.join(root,'.chrome-profile-cdp-'+process.pid);
const browser=process.argv[2];
if(!browser)throw Error('Chromium executable path required.');
const log=fs.openSync(path.join(root,'results/browser_cdp.log'),'w');
const proc=spawn(browser,['--headless','--no-sandbox','--disable-gpu','--disable-software-rasterizer','--in-process-gpu','--no-first-run','--disable-extensions','--disable-background-networking','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore',log],windowsHide:true});
let ws;
(async()=>{
 const portfile=path.join(profile,'DevToolsActivePort');
 for(let i=0;i<100&&!fs.existsSync(portfile);i++)await sleep(50);
 if(!fs.existsSync(portfile))throw Error('DevTools port not created.');
 const port=Number(fs.readFileSync(portfile,'utf8').split(/\r?\n/)[0]);
 const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 const target=targets.find(t=>t.type==='page');
 ws=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 let seq=0;const pending=new Map();
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}});
 function send(method,params={}){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await send('Page.enable');await send('Runtime.enable');
 async function navigate(file){await send('Page.navigate',{url:pathToFileURL(file).href});for(let i=0;i<100;i++){if(await evaluate('document.readyState')==='complete')return;await sleep(30);}throw Error('Page load timeout');}
 const results=[];
 for(const [name,width,height] of [['desktop',1440,1100],['mobile',390,844]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await navigate(path.join(root,'tests/browser_smoke.html'));
  for(let i=0;i<150;i++){const title=await evaluate('document.title');if(title==='BROWSER_QA_PASS'||title==='BROWSER_QA_FAIL')break;await sleep(30);}
  const text=await evaluate('document.querySelector("#browser-qa-result")?.textContent');
  const result=JSON.parse(text||'{}');results.push(result);
  if(result.status!=='PASS')throw Error(JSON.stringify(result));
  console.log(name,result.status,result.checks);
  await navigate(path.join(root,'web/dist/index.html'));await sleep(250);
  const image=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  fs.writeFileSync(path.join(root,'results/web_'+name+'.png'),Buffer.from(image.data,'base64'));
 }
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await navigate(path.join(root,'web/dist/index.html'));await sleep(150);
 await evaluate('document.documentElement.style.scrollBehavior="auto";document.querySelector("#lab").scrollIntoView({behavior:"instant"})');await sleep(100);
 const lab=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 fs.writeFileSync(path.join(root,'results/web_lab.png'),Buffer.from(lab.data,'base64'));
 fs.writeFileSync(path.join(root,'results/browser_qa.json'),JSON.stringify(results,null,2));
 await send('Browser.close');
 console.log('Browser QA results and screenshots saved.');
})().catch(error=>{console.error(error.message);process.exitCode=1;}).finally(()=>{if(ws)ws.close();if(proc.exitCode===null)proc.kill();fs.closeSync(log);});
