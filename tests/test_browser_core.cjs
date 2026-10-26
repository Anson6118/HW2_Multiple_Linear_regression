'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
require('../web/dist/app.js');
const c=globalThis.HW2Core;
const n=120,p=3;
const raw=Array.from({length:n},(_,i)=>[Math.sin(i*.31),Math.cos(i*.73),Math.sin(i*.17+1)]);
const means=Array.from({length:p},(_,j)=>raw.reduce((s,r)=>s+r[j],0)/n);
const scales=Array.from({length:p},(_,j)=>Math.sqrt(raw.reduce((s,r)=>s+(r[j]-means[j])**2,0)/n));
const X=raw.map(r=>r.map((v,j)=>(v-means[j])/scales[j]));
const y=X.map((r,i)=>2+3*r[0]-.8*r[1]+.2*r[2]+Math.cos(i)*.01);
if(process.argv.includes('--reference')){
  const fitted={};for(const name of ['LinearRegression','Ridge','Lasso']){const m=c.fitModel(X,y,name,.3);fitted[name]={coef:m.coef,intercept:m.intercept};}
  console.log(JSON.stringify({X,y,fitted}));process.exit(0);
}
let checks=0;
function test(name,fn){fn();checks++;console.log('PASS '+name);}
test('CSV quoted commas, doubled quotes, multiline, BOM and CRLF',()=>{
 const r=c.parseCSV('\uFEFFx,label,y\r\n1,"a,b",2\r\n3,"a""b\nnext",4\r\n');
 assert.deepEqual(r.headers,['x','label','y']);assert.equal(r.rows[1][1],'a"b\nnext');assert.equal(r.rows.length,2);
});
test('Reject malformed CSV and duplicate headers',()=>{assert.throws(()=>c.parseCSV('a,a\n1,2'));assert.throws(()=>c.parseCSV('a,b\n1'));assert.throws(()=>c.parseCSV('a,b\n1,"x'));});
test('OLS recovers analytic coefficients',()=>{const target=X.map(r=>1+2*r[0]-3*r[1]+.5*r[2]);const m=c.fitModel(X,target);assert.ok(Math.abs(m.intercept-1)<1e-12);[2,-3,.5].forEach((v,j)=>assert.ok(Math.abs(m.coef[j]-v)<1e-10));});
test('Ridge reduces coefficient norm',()=>{const a=c.fitModel(X,y),b=c.fitModel(X,y,'Ridge',10);assert.ok(b.coef.reduce((s,v)=>s+v*v,0)<a.coef.reduce((s,v)=>s+v*v,0));});
test('Lasso high penalty zeroes coefficients',()=>{const m=c.fitModel(X,y,'Lasso',100);assert.ok(m.coef.every(v=>v===0));});
test('Metrics MAPE zero handling, adjusted R2 undefined for small n',()=>{const m=c.score([0,2],[1,1],5);assert.equal(m.MAPE,50);assert.equal(m.MAPE_excluded_zeros,1);assert.equal(m.Adjusted_R2,null);});
test('Both example datasets train with categorical encoding',()=>{
 for(const [file,target,p] of [['flood_sample.csv','FloodProbability',20],['50_Startups.csv','Profit',5]]){
  const r=c.trainCSV(c.parseCSV(fs.readFileSync(path.join(__dirname,'../web/dist',file),'utf8')),target,'Ridge','all',.01);
  assert.equal(r.features.length,p);assert.ok(Number.isFinite(r.metrics.RMSE));assert.equal(r.split.train+r.split.validation+r.split.test,file==='flood_sample.csv'?2000:50);
 }
});
test('Deterministic splits; held-out extremes do not affect training scaling',()=>{
 const rows=Array.from({length:100},(_,i)=>[String(i),String(i*2+1)]),d={headers:['x','y'],rows};
 const before=c.trainCSV(d,'y','Ridge','all',.1),ids=c.shuffleIndices(100),modified=structuredClone(d);
 for(const i of ids.slice(60))modified.rows[i][0]='1000000';
 const after=c.trainCSV(modified,'y','Ridge','all',.1);
 assert.deepEqual(before.means,after.means);assert.deepEqual(before.scales,after.scales);assert.deepEqual(before.coefficients,after.coefficients);
 const disjoint=new Set(ids);assert.equal(disjoint.size,100);
});
test('Unknown validation/test category does not enter fitted schema',()=>{
 const d={headers:['x','group','y'],rows:Array.from({length:100},(_,i)=>[String(i),i%2?'A':'B',String(i*2+1)])};
 for(const i of c.shuffleIndices(100).slice(60))d.rows[i][1]='UNSEEN';
 const r=c.trainCSV(d,'y','Ridge','all',1);assert.ok(!r.schema.find(s=>s.name==='group').levels.includes('UNSEEN'));assert.ok(Number.isFinite(r.metrics.RMSE));
});
test('Training medians handle missing features, invalid targets are omitted',()=>{
 const d={headers:['x','y'],rows:Array.from({length:50},(_,i)=>[i%7?'':String(i),String(i+2)])};d.rows[0][1]='';
 const r=c.trainCSV(d,'y','Ridge','all',1);assert.equal(r.omitted,1);assert.ok(Number.isFinite(r.metrics.RMSE));
});
test('Reject invalid alpha and insufficient numeric targets',()=>{assert.throws(()=>c.trainCSV({headers:['x','y'],rows:[['1','2']]},'y'));const d={headers:['x','y'],rows:Array.from({length:30},(_,i)=>[String(i),String(i)])};assert.throws(()=>c.trainCSV(d,'y','Lasso','all',0));assert.throws(()=>c.trainCSV(d,'y','Ridge','all',NaN));});
console.log(`${checks} browser-core checks passed.`);
