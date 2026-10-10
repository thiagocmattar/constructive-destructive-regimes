// Headless Chrome QA using Node's native WebSocket/CDP; isolated ignored profile.
import {spawn} from 'node:child_process';
import {writeFile,mkdir,readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const here=path.dirname(fileURLToPath(import.meta.url));
const scratch=path.join(here,'scratch');
await mkdir(scratch,{recursive:true});
const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile=path.join(scratch,'chrome-profile-'+process.pid);
const browser=spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:'ignore'});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket;
try {
 let targets;
 for(let i=0;i<50;i++){try{const port=(await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();break;}catch{await pause(100);}}
 if(!targets)throw new Error('Headless Chrome did not start');
 const target=targets.find(t=>t.type==='page');
 socket=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 const waiting=new Map(),errors=[];let sequence=0;
 socket.addEventListener('close',event=>{for(const pending of waiting.values())pending.reject(new Error(`CDP closed: ${event.code} ${event.reason}`));waiting.clear();});
 socket.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const p=waiting.get(message.id);if(p){waiting.delete(message.id);message.error?p.reject(message.error):p.resolve(message.result);}}else if(message.method==='Runtime.exceptionThrown'){errors.push(message.params.exceptionDetails);}});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{waiting.delete(id);reject(new Error('CDP timeout: '+method));},10000);waiting.set(id,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:error=>{clearTimeout(timer);reject(error);}});socket.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
 await send('Runtime.enable');await send('Page.enable');
 await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'light'}]});
 await send('Emulation.setDeviceMetricsOverride',{width:1200,height:1900,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:8765/rosenbrock-explorer.html'});
 let ready=false;
 for(let i=0;i<80;i++){ready=await evaluate("document.getElementById('rosenbrock-pilot')?.dataset.ready==='true'");if(ready)break;await pause(100);}
 if(!ready)throw new Error('Explorer failed to render');
 const initial=await evaluate("({ready:document.querySelector('main').dataset.ready,step:document.querySelector('main').dataset.step,detail:document.getElementById('detail').textContent,canvases:[...document.querySelectorAll('canvas')].map(c=>({id:c.id,width:c.width,height:c.height})),overflow:document.documentElement.scrollWidth>innerWidth})");
 if(initial.overflow||initial.canvases.length!==8||initial.canvases.some(c=>c.width===0))throw new Error('Initial layout failure');
 const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(path.join(scratch,'explorer-desktop.png'),Buffer.from(desktop.data,'base64'));
 const checks=[];
 for(let condition=0;condition<4;condition++){
  for(const stream of ['raw','momentum','adaptive','displacement']){
   const actual=await evaluate(`(()=>{const c=document.getElementById('condition');c.value='${condition}';c.dispatchEvent(new Event('change'));const s=document.getElementById('stream');s.value='${stream}';s.dispatchEvent(new Event('change'));const t=document.getElementById('update');t.value='4321';t.dispatchEvent(new Event('change'));document.getElementById('previous').click();return {condition:document.querySelector('main').dataset.condition,stream:document.querySelector('main').dataset.stream,step:document.querySelector('main').dataset.step,detail:document.getElementById('detail').textContent};})()`);
   if(actual.condition!==String(condition)||actual.stream!==stream||actual.step!=='4320')throw new Error('Linked interaction failure');
   checks.push(actual);
  }
 }
 const undefinedCases=await evaluate("(()=>{const c=document.getElementById('condition');c.value='0';c.dispatchEvent(new Event('change'));const s=document.getElementById('stream');s.value='raw';s.dispatchEvent(new Event('change'));const t=document.getElementById('update');t.value='0';t.dispatchEvent(new Event('change'));const zero=document.getElementById('detail').textContent;t.value='1';t.dispatchEvent(new Event('change'));return {zero,one:document.getElementById('detail').textContent};})()");
 if(!undefinedCases.zero.includes('undefined')||!undefinedCases.one.includes('undefined'))throw new Error('Undefined first cosine hidden');
 await evaluate("(()=>{const t=document.getElementById('update');t.value='9990';t.dispatchEvent(new Event('change'));document.getElementById('local-time').click();document.getElementById('zoom').click();})()");
 const zoom=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(path.join(scratch,'explorer-zoom.png'),Buffer.from(zoom.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:360,height:1100,deviceScaleFactor:1,mobile:true});
 await pause(300);
 const narrow=await evaluate("({overflow:document.documentElement.scrollWidth>innerWidth,width:innerWidth,canvases:[...document.querySelectorAll('canvas')].map(c=>c.clientWidth)})");
 if(narrow.overflow||narrow.canvases.some(w=>w>360))throw new Error('Narrow layout overflow');
 const mobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(path.join(scratch,'explorer-mobile.png'),Buffer.from(mobile.data,'base64'));
 await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'dark'}]});
 await pause(250);
 const dark=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(path.join(scratch,'explorer-dark.png'),Buffer.from(dark.data,'base64'));
 if(errors.length)throw new Error(JSON.stringify(errors));
 await writeFile(path.join(here,'ui_verification.json'),JSON.stringify({passed:true,checked_utc:new Date().toISOString(),
  initial,condition_stream_checks:checks.length,selected_update_and_previous:true,undefinedCases,narrow,
  local_window_and_transition_zoom:true,light_and_dark_screenshots:true,uncaught_exceptions:errors,
  screenshots:'ignored scratch/explorer-{desktop,zoom,mobile,dark}.png'},null,2)+'\n');
 console.log(JSON.stringify({passed:true,condition_stream_checks:checks.length,narrow_width:narrow.width,uncaught_errors:errors.length}));
 await send('Browser.close');
} finally {
 socket?.close();browser.kill();
}
