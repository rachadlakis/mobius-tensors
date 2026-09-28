const {chromium}=require('playwright');const fs=require('fs');
const m=JSON.parse(fs.readFileSync('brand/_manifest.json'));
(async()=>{const b=await chromium.launch();fs.mkdirSync('brand/png',{recursive:true});
for(const [k,t] of Object.entries(m)){const s=fs.readFileSync(`brand/${k}.svg`,'utf8');
 const w=+s.match(/width="(\d+)"/)[1],h=+s.match(/height="(\d+)"/)[1];
 const scales = k.startsWith('app-icon')||k.startsWith('favicon') ? [1] : k.startsWith('linkedin')||k.startsWith('website') ? [1,2] : [2];
 for(const sc of scales){const p=await b.newPage({viewport:{width:w,height:h},deviceScaleFactor:sc});
  await p.setContent(`<html><body style="margin:0;background:transparent">${s}</body></html>`);
  const name = scales.length>1 ? `${k}${sc==2?'@2x':''}` : k;
  await p.screenshot({path:`brand/png/${name}.png`,omitBackground:t,clip:{x:0,y:0,width:w,height:h}});await p.close();}}
await b.close()})();
