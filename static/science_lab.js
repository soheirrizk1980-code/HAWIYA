 'use strict';
// This script adds scientific interpretations to HAWIYA without modifying AI inference.
(function(){
 const node=id=>document.getElementById(id);
 const set=(id,value)=>{const el=node(id);if(el)el.textContent=value;};
 const numeric=value=>Number.isFinite(Number(value))?Number(value):null;
 const fmt=(value,digits=2)=>numeric(value)===null?'—':Number(value).toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits});
 const ctxOf=id=>node(id).getContext('2d',{willReadFrequently:true});
 let previewImage=new Image(), stripImage=new Image(), loadedImage='',loadedStrip='';
 let lightReady=false,stripReady=false;
 function loadImage(url,kind){
  if(!url)return;
  if(kind==='photo' && loadedImage===url)return;
  if(kind==='strip' && loadedStrip===url)return;
  const im=kind==='photo'?previewImage:stripImage;
  if(kind==='photo'){loadedImage=url;lightReady=false;}else{loadedStrip=url;stripReady=false;}
  im.onload=()=>{if(kind==='photo'){lightReady=true;drawLight();}else{stripReady=true;drawGamma();}};
  im.onerror=()=>{if(kind==='photo')lightReady=false;else stripReady=false;};
  im.src=url;
 }
 function drawLight(){
  if(!lightReady)return;
  const c=node('lightPreview'),x=c.getContext('2d',{willReadFrequently:true}),shift=Number(node('lightOffset').value);
  x.clearRect(0,0,c.width,c.height);x.drawImage(previewImage,0,0,c.width,c.height);
  const frame=x.getImageData(0,0,c.width,c.height),a=frame.data,hist=Array(256).fill(0);let sum=0;
  for(let i=0;i<a.length;i+=4){
   const r=Math.max(0,Math.min(255,a[i]+shift)),g=Math.max(0,Math.min(255,a[i+1]+shift)),b=Math.max(0,Math.min(255,a[i+2]+shift));
   a[i]=r;a[i+1]=g;a[i+2]=b;
   const y=Math.max(0,Math.min(255,Math.round(.299*r+.587*g+.114*b)));hist[y]++;sum+=y;
  }
  x.putImageData(frame,0,0);
  const h=node('histogramCanvas'),hc=h.getContext('2d'),max=Math.max(...hist,1);
  hc.clearRect(0,0,h.width,h.height);hc.fillStyle='#152a28';hc.fillRect(0,0,h.width,h.height);
  hc.fillStyle='#7bd0ab';for(let b=0;b<256;b++)hc.fillRect(b*h.width/256,h.height-8-(hist[b]/max)*(h.height-18),Math.max(1,h.width/256),hist[b]/max*(h.height-18));
  set('lightOut',(shift>0?'+':'')+shift);set('lightMetrics','Preview mean: '+fmt(sum/(a.length/4),1)+' / 255 · live 256-bin histogram');
 }
 function drawGamma(){
  if(!stripReady)return;
  const c=node('gammaPreview'),ctx=c.getContext('2d',{willReadFrequently:true});ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(stripImage,0,0,c.width,c.height);
  const frame=ctx.getImageData(0,0,c.width,c.height),a=frame.data,gamma=Number(node('gammaLab').value);
  for(let i=0;i<a.length;i+=4)for(let j=0;j<3;j++)a[i+j]=Math.round(255*Math.pow(a[i+j]/255,gamma));
  ctx.putImageData(frame,0,0);set('gammaOut',gamma.toFixed(2));set('gammaMetrics','Browser-only gamma preview · model inputs and embeddings are unchanged');
 }
 function updateModelNumbers(){
  if(typeof analysis==='undefined'||!analysis||analysis.status!=='detected'){
    set('dataOrigin','Select an eye and press Analyze to populate these measurements. Sample preview values are shown only when cached model output has loaded.');return;
  }
  const q=analysis.quality||{},g=analysis.geometry||{},ir=g.iris,pu=g.pupil,confidence=analysis.confidence||{};
  const cache=!!(eyeA&&eyeA.sample&&preparedCache?.analyses?.[eyeA.sample]===analysis);
  set('dataOrigin',cache?'SOURCE: Precomputed results from the trained YOLO + ArcIris models. Press Analyze to recompute live.':'SOURCE: Backend model analysis. Browser experiments below are calculated independently.');
  set('mResolution',q.width&&q.height?`${q.width} × ${q.height}`:'—');
  set('mBrightness',fmt(q.brightness,1));set('mFocus',fmt(q.sharpness,1));
  set('mRatio',ir&&pu?fmt(pu[2]/ir[2],3):'—');
  set('mArea',ir&&pu?fmt(Math.PI*(ir[2]**2-pu[2]**2),0):'—');
  set('mFeatures',analysis.embedding_dimensions===512?'512 values':'—');
  set('mConf',numeric(confidence.iris)!==null&&numeric(confidence.pupil)!==null?`${fmt(confidence.iris*100,1)}% / ${fmt(confidence.pupil*100,1)}%`:'—');
  set('mIrisCenter',ir?`${fmt(ir[0],1)}, ${fmt(ir[1],1)} px`:'—');
  set('mPupilCenter',pu?`${fmt(pu[0],1)}, ${fmt(pu[1],1)} px`:'—');
  set('mRadii',ir&&pu?`${fmt(ir[2],1)} / ${fmt(pu[2],1)} px`:'—');
  if(analysis.crop)loadImage(analysis.crop,'photo');
  if(analysis.normalized_gray)loadImage(analysis.normalized_gray,'strip');
  const bars=node('vectorBars');bars.replaceChildren();
  const features=analysis.feature_profile;
  if(Array.isArray(features)&&features.length){
   const max=Math.max(.00001,...features.map(v=>Math.abs(v)));
   features.forEach((v,i)=>{const bar=document.createElement('i');bar.style.height=Math.max(3,Math.abs(v)/max*100)+'%';bar.className=v<0?'negative':'';bar.title=`Group ${i+1} mean: ${fmt(v,6)}`;bars.append(bar);});
  }
  const similarity=typeof comparison!=='undefined'&&comparison?.similarity;
  set('mGrayScore',similarity?fmt(similarity.grayscale,5):'—');set('mGammaScore',similarity?fmt(similarity.gamma,5):'—');
 }
 function updateExperimentNumbers(){
  if(typeof getGeometry!=='function'||typeof useFitted==='undefined'||!useFitted||!sourceGeom||!imageReady){
    set('mAreaDelta','—');set('mStripDifference','—');return;
  }
  const current=getGeometry();if(!current)return;
  const a=node('inner'),b=node('outer'),va=a.value,vb=b.value;
  // Obtain zero-offset geometry without recomputing backend data.
  let reference;try{a.value='0';b.value='0';reference=getGeometry();}finally{a.value=va;b.value=vb;}
  if(!reference)return;
  const area=g=>Math.PI*Math.max(0,g.iris.r*g.iris.r-g.pupil.r*g.pupil.r);
  set('mAreaDelta',`${fmt((area(current)/Math.max(1,area(reference))-1)*100,2)}% vs 0 px`);
  const original=ctx.getImageData(0,0,off.width,off.height).data;
  function point(g,t,r){return {x:(1-r)*(g.pupil.x+g.pupil.r*Math.cos(t))+r*(g.iris.x+g.iris.r*Math.cos(t)),y:(1-r)*(g.pupil.y+g.pupil.r*Math.sin(t))+r*(g.iris.y+g.iris.r*Math.sin(t))};}
  let sq=0,n=0;
  for(let row=0;row<16;row++)for(let col=0;col<64;col++){
   const radius=(row+.5)/16,angle=col*Math.PI*2/64;
   const p=point(current,angle,radius),r=point(reference,angle,radius);
   const c1=sampleRGB(original,p.x,p.y),c2=sampleRGB(original,r.x,r.y);
   for(let ch=0;ch<3;ch++){sq+=(c1[ch]-c2[ch])**2;n++;}
  }
  set('mStripDifference',fmt(Math.sqrt(sq/n),2)+' / 255 sampled RMS');
 }
 function refresh(){updateModelNumbers();updateExperimentNumbers();}
 window.refreshScience=refresh;
 node('lightOffset').addEventListener('input',drawLight);
 node('gammaLab').addEventListener('input',drawGamma);
 refresh();
})();
