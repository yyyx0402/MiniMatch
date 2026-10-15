const $ = selector => document.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names = {hybrid:'智能搜索', minilm:'语义搜索', keyword:'关键词搜索'};
const modeNotes = {hybrid:'结合关键词与语义，更全面地寻找相关商品', minilm:'理解需求与用途，寻找语义相关的商品', keyword:'按商品文字匹配名称、品牌与型号'};
const selected = new Map(), products = new Map();
let model='hybrid', searchGeneration=0, controller=null, detailGeneration=0, detailProduct=null, activeQuery='', offset=0, hasMore=false;
const storage = {
  read(key, fallback) {try {return JSON.parse(localStorage.getItem('productsearch.'+key)) ?? fallback;} catch {return fallback;}},
  write(key, value) {try {localStorage.setItem('productsearch.'+key, JSON.stringify(value));} catch {}}
};
let recent = storage.read('recent', []);
if (!Array.isArray(recent)) recent=[];
recent=recent.filter(q=>typeof q==='string' && q.length<=240).slice(0,6);
async function api(url, signal) {
  const response=await fetch(url, {signal});
  const data=await response.json();
  if (!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : '请求失败，请检查输入后重试。');
  return data;
}
function switchTab(tab) {
  if (!['search','compare','about'].includes(tab)) return;
  for (const name of ['search','compare','about']) $('#'+name+'-view').hidden=name!==tab;
  $('#page-name').textContent={search:'发现商品',compare:'商品对比',about:'使用说明'}[tab];
  document.querySelectorAll('.nav-item').forEach(b=>{b.classList.toggle('active',b.dataset.tab===tab);b.setAttribute('aria-current',b.dataset.tab===tab?'page':'false');});
  if (tab==='compare') renderComparison();
  $('#compare-tray').hidden=!selected.size || tab!=='search';
  window.scrollTo({top:0,behavior:'smooth'});
}
function recentSearches() {
  $('#recent-searches').hidden=!recent.length;
  $('#recent-searches').innerHTML='<span>最近搜索</span>'+recent.map(q=>`<button data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join('')+'<button class="text-button" id="clear-recent">清除</button>';
}
function card(p) {
  return `<article class="product-card"><div class="card-top"><span class="rank">${String(p.rank).padStart(2,'0')} / RESULT</span><span class="card-brand">${escapeHtml(p.brand || '品牌未提供')}</span></div><h2>${escapeHtml(p.title)}</h2><p class="card-description">${escapeHtml(p.bullets || p.description || '未提供详细描述。')}</p><div class="card-tags">${p.color?`<span class="tag">${escapeHtml(p.color.slice(0,50))}</span>`:''}${p.matched_terms.slice(0,5).map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}</div><div class="card-bottom"><button class="text-button details-button" data-detail="${escapeHtml(p.id)}">查看详情 ↗</button><button class="compare-button ${selected.has(p.id)?'chosen':''}" data-compare="${escapeHtml(p.id)}" aria-pressed="${selected.has(p.id)}">${selected.has(p.id)?'✓ 已加入':'＋ 加入对比'}</button></div></article>`;
}
async function search(append=false) {
  const query=append?activeQuery:$('#query').value.trim();
  const generation=++searchGeneration;
  controller?.abort();controller=new AbortController();
  const requestedModel=model, requestedOffset=append?offset:0;
  $('#search-error').hidden=true;$('#search-button').disabled=true;$('#load-more').disabled=true;$('#load-more').hidden=true;$('#results').setAttribute('aria-busy','true');
  if (!append) {products.clear();$('#results').innerHTML='';hasMore=false;}
  $('#search-status').textContent=append?'正在加载更多商品…':'正在搜索商品，首次加载模型需要稍等片刻…';
  try {
    const data=await api(`/api/search?q=${encodeURIComponent(query)}&model=${requestedModel}&offset=${requestedOffset}`,controller.signal);
    if (generation!==searchGeneration) return;
    activeQuery=data.query;
    data.results.forEach(p=>products.set(p.id,p));
    $('#results').insertAdjacentHTML('beforeend',data.results.map(card).join(''));
    if (!products.size) $('#results').innerHTML='<div class="empty">没有找到匹配商品。试试减少限定词，或切换智能搜索。</div>';
    offset=requestedOffset+data.results.length;hasMore=data.has_more;
    $('#search-status').textContent=`“${data.query}” · 已展示 ${products.size} 件商品 · ${names[data.model]}`;
    if (!append) {recent=[data.query,...recent.filter(q=>q.toLowerCase()!==data.query.toLowerCase())].slice(0,6);storage.write('recent',recent);recentSearches();}
    updateSelection(false);
  } catch(error) {
    if (generation!==searchGeneration || error.name==='AbortError') return;
    $('#search-status').textContent=append?`已展示 ${products.size} 件商品`:'';
    $('#search-error').textContent=error.message;$('#search-error').hidden=false;
  } finally {
    if (generation===searchGeneration) {$('#search-button').disabled=false;$('#load-more').disabled=false;$('#load-more').hidden=!hasMore;$('#results').setAttribute('aria-busy','false');}
  }
}
function updateSelection(persist=true) {
  $('#compare-count').textContent=selected.size;$('#nav-count').textContent=selected.size;$('#compare-tray').hidden=!selected.size || $('#search-view').hidden;
  document.querySelectorAll('[data-compare]').forEach(b=>{const chosen=selected.has(b.dataset.compare);b.classList.toggle('chosen',chosen);b.textContent=chosen?'✓ 已加入':'＋ 加入对比';b.setAttribute('aria-pressed',String(chosen));});
  if (persist) storage.write('compare',[...selected.keys()]);
  renderComparison();
}
function renderComparison() {
  $('#compare-empty').hidden=!!selected.size;$('#compare-content').hidden=!selected.size;
  if (!selected.size) {$('#compare-table').innerHTML='';return;}
  $('#compare-help').textContent=selected.size===1?'已选一件商品，还可以添加两件。':`正在比较 ${selected.size} 件商品`;
  const items=[...selected.values()];
  const fields=[['商品名称','title'],['品牌','brand'],['颜色','color'],['产品要点','bullets'],['商品描述','description'],['商品 ID','id']];
  $('#compare-table').innerHTML=`<table><thead><tr><th>商品信息</th>${items.map((p,i)=>`<th>商品 ${i+1}<button class="remove-product" data-remove="${escapeHtml(p.id)}" aria-label="移除商品 ${i+1}">移除</button></th>`).join('')}</tr></thead><tbody>${fields.map(([label,key])=>`<tr><th>${label}</th>${items.map(p=>`<td><p>${escapeHtml(p[key]||'未提供')}</p></td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
async function openDetail(id) {
  const generation=++detailGeneration;
  detailProduct=null;$('#detail-content').textContent='正在读取商品信息…';$('#detail-dialog').showModal();
  try {
    const p=await api('/api/products/'+encodeURIComponent(id));
    if (generation!==detailGeneration || !$('#detail-dialog').open) return;
    detailProduct=p;
    $('#detail-content').innerHTML=`<h3 class="detail-title">${escapeHtml(p.title)}</h3><div class="detail-meta"><span>品牌：${escapeHtml(p.brand||'未提供')}</span><span>颜色：${escapeHtml(p.color||'未提供')}</span></div><h4>产品要点</h4><p class="detail-text">${escapeHtml(p.bullets||'未提供')}</p><h4>商品描述</h4><p class="detail-text">${escapeHtml(p.description||'未提供')}</p><div class="detail-bottom"><small>商品 ID：${escapeHtml(p.id)}</small><button class="compare-button" data-compare="${escapeHtml(p.id)}">加入对比</button></div>`;
    updateSelection(false);
  } catch(error) {if(generation===detailGeneration)$('#detail-content').textContent=error.message;}
}
document.addEventListener('click',e=>{
  const tab=e.target.closest('[data-tab]');if(tab){switchTab(tab.dataset.tab);return;}
  const suggestion=e.target.closest('[data-query]');if(suggestion){$('#query').value=suggestion.dataset.query;switchTab('search');search();return;}
  const mode=e.target.closest('[data-model]');if(mode){model=mode.dataset.model;document.querySelectorAll('[data-model]').forEach(b=>{b.classList.toggle('selected',b===mode);b.setAttribute('aria-pressed',String(b===mode));});$('#mode-note').textContent=modeNotes[model];search();return;}
  const detail=e.target.closest('[data-detail]');if(detail){openDetail(detail.dataset.detail);return;}
  const remove=e.target.closest('[data-remove]');if(remove){selected.delete(remove.dataset.remove);updateSelection();return;}
  const compare=e.target.closest('[data-compare]');if(compare){
    const id=compare.dataset.compare;
    if(selected.has(id))selected.delete(id);
    else if(selected.size<3){const product=products.get(id) || (detailProduct?.id===id?detailProduct:null);if(product)selected.set(id,product);}
    else {const message='最多比较三件商品，请先移除一件。';if($('#detail-dialog').open){let note=$('#detail-limit');if(!note){note=document.createElement('p');note.id='detail-limit';note.className='small-note';$('#detail-content').append(note);}note.textContent=message;}else $('#search-status').textContent=message;return;}
    updateSelection();return;
  }
  if(e.target.closest('#clear-recent')){recent=[];storage.write('recent',recent);recentSearches();}
});
$('#search-form').addEventListener('submit',e=>{e.preventDefault();search();});
$('#load-more').addEventListener('click',()=>search(true));
for (const id of ['clear-compare','tray-clear']) $('#'+id).addEventListener('click',()=>{selected.clear();updateSelection();});
$('#open-compare').addEventListener('click',()=>switchTab('compare'));
$('#close-detail').addEventListener('click',()=>$('#detail-dialog').close());
$('#detail-dialog').addEventListener('close',()=>{++detailGeneration;detailProduct=null;});
async function init() {
  recentSearches();
  try {const info=await api('/api/application');$('#connection').textContent='本地搜索已就绪';$('#catalog-note').textContent=`${info.catalog_products.toLocaleString()} 件商品 · 英文搜索`;}
  catch(error){$('#connection').textContent='搜索服务暂不可用';$('#search-error').hidden=false;$('#search-error').textContent=error.message;$('#search-status').textContent='';return;}
  const saved=storage.read('compare',[]);
  if (Array.isArray(saved)) {
    const ids=[...new Set(saved.filter(id=>typeof id==='string'))].slice(0,3);
    const results=await Promise.allSettled(ids.map(id=>api('/api/products/'+encodeURIComponent(id))));
    results.forEach(r=>{if(r.status==='fulfilled' && selected.size<3)selected.set(r.value.id,r.value);});
    updateSelection();
  }
  search();
}
init();
