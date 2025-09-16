// model.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { uploadModelArtifactsToFirebase } from './firebase.js';

/*
  Model & training design notes (browser-friendly):
  - Small Transformer-ish architecture with embedding, simple self-attention and dense heads.
  - Sequence length (SEQ_LEN) is fixed; model predicts a next-token distribution.
  - generateWithWebBlend: samples multiple tokens autoregressively to create a reply,
    simultaneously runs a web search (DuckDuckGo via AllOrigins proxy), and blends snippet.
  - applyRewardUpdate: REINFORCE-like update computed in-browser (one optimizer step).
  - supervisedTrainOnTexts: light supervised fine-tuning on crawled text samples.
  - Persistence: load/save from IndexedDB (tf.io.browserIndexedDB). Also exposes upload to Firebase Storage.
*/

const SEQ_LEN = 12;
const MAX_GEN = 12;
const EMBED_DIM = 96;
const ATT_DIM = 96;
const HIDDEN = 192;
const LR = 3e-4;

let globalVocab = null; // word->idx
let globalInvVocab = null; // idx->word
let optimizer = tf.train.adam(LR);

// Utility: load vocabulary JSON from vocab.json at root (must exist)
async function loadVocab(){
  if(globalVocab) return { vocab: globalVocab, invVocab: globalInvVocab };
  const res = await fetch('./vocab.json');
  const j = await res.json();
  globalVocab = j;
  globalInvVocab = Object.fromEntries(Object.entries(j).map(([k,v])=>[v,k]));
  return { vocab: globalVocab, invVocab: globalInvVocab };
}

// Build model and return {model, vocab, invVocab}
export async function createModel(){
  const { vocab, invVocab } = await loadVocab();
  const vocabSize = Object.keys(vocab).length;

  const input = tf.input({shape:[SEQ_LEN], dtype:'int32', name:'tokens'});
  const embed = tf.layers.embedding({ inputDim: vocabSize, outputDim: EMBED_DIM }).apply(input); // [b,SEQ,EMB]

  // simple projection q,k,v
  const q = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embed);
  const k = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embed);
  const v = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embed);

  // scores and softmax (scaled)
  const scores = tf.layers.dot({axes:[2,2]}).apply([q,k]); // [b,SEQ,SEQ]
  const scaled = tf.layers.lambda(x => tf.mul(x, 1/Math.sqrt(ATT_DIM))).apply(scores);
  const weights = tf.layers.activation({activation:'softmax'}).apply(scaled);
  const context = tf.layers.dot({axes:[2,1]}).apply([weights, v]); // [b,SEQ,ATT_DIM]

  const flat = tf.layers.flatten().apply(context);
  const d1 = tf.layers.dense({units:HIDDEN, activation:'relu'}).apply(flat);
  const logits = tf.layers.dense({units:vocabSize}).apply(d1); // logits for next token

  const model = tf.model({inputs: input, outputs: logits});
  // we won't compile with a loss because we do custom optimize for reward; for supervised training we'll use model.fit
  return { model, vocab, invVocab };
}

// IndexedDB persistence helpers
export async function saveModelToIndexedDB(model, name='xylxx-model'){
  try{
    await model.save(`indexeddb://${name}`);
    console.log('Model saved to IndexedDB');
  }catch(e){ console.error('saveModelToIndexedDB failed', e); throw e; }
}

export async function initModelFromIndexedDB(name='xylxx-model'){
  try{
    // test if model exists
    const models = await tf.io.listModels();
    const key = `indexeddb://${name}`;
    if(models[key]){
      const model = await tf.loadLayersModel(key);
      const { vocab, invVocab } = await loadVocab();
      return { model, vocab, invVocab };
    }
    return null;
  }catch(e){ console.warn('initModelFromIndexedDB error', e); return null; }
}

// ----------------- Token helpers -----------------
export function textToIds(text, vocab){
  const words = (text || '').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
  const ids = words.map(w => (w in vocab) ? vocab[w] : 0);
  // left pad with zeros
  const pad = Math.max(0, SEQ_LEN - ids.length);
  return new Array(pad).fill(0).concat(ids.slice(-SEQ_LEN));
}

export function idsToText(ids, invVocab){
  return ids.map(i => invVocab[i] ?? '…').join(' ');
}

// ----------------- Generation -----------------
/**
 * generateWithWebBlend:
 * - sample tokens autoregressively using model
 * - in parallel call webSearch(query) to fetch a snippet
 * - return reply text blended from generated tokens and snippet
 */
export async function generateWithWebBlend(model, promptText, temperature=1.0){
  const { vocab, invVocab } = await loadVocab();
  // prepare context
  let ctx = textToIds(promptText, vocab);

  // sample tokens
  const sampled = [];
  for(let step=0; step<MAX_GEN; step++){
    const inputTensor = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
    const logits = model.predict(inputTensor); // [1, vocab]
    const logits1d = logits.squeeze(); // [vocab]
    const probs = tf.softmax(tf.div(logits1d, tf.scalar(temperature)));
    const probsData = await probs.data();
    const idx = sampleFromProbs(probsData);
    sampled.push(idx);
    ctx = ctx.slice(1).concat([idx]);
    tf.dispose([inputTensor, logits, logits1d, probs]);
    // tiny break if sample is a padding (0) often we can continue though
  }

  const genText = idsToText(sampled, invVocab);

  // start web search in parallel
  const webSnippet = await webSearchSnippet(promptText);

  // create blended reply
  let reply = genText;
  if(webSnippet && webSnippet.trim().length > 20){
    reply = `${genText}\n\n(From web): ${webSnippet}`;
  } else {
    // if snippet too short, try short explanatory fallback
    if(!genText || genText.trim().length < 2) reply = "I'm not sure — I couldn't find a good answer.";
  }

  const turnId = 't_' + Date.now().toString(36) + '_' + Math.floor(Math.random()*9999).toString(36);

  // Also kick off lightweight learning from web pages asynchronously: crawl and supervised update
  if(webSnippet){
    (async ()=>{
      try{
        const pages = await crawlPagesForQuery(promptText, 2); // fetch 2 pages
        // train lightly on these pages (no await so UI is fast)
        await supervisedTrainOnTexts(model, pages.map(p=>p.text), 1);
        await saveModelToIndexedDB(model);
      }catch(e){ console.warn('background crawl/train failed', e); }
    })();
  }

  // Build prompt tokens for reward support
  const promptTokens = textToIds(promptText, globalVocab || (await loadVocab()).vocab);
  return { reply, promptTokens, sampledTokens: sampled, turnId };
}

// basic sampling helper
function sampleFromProbs(probs){
  // normalize
  let sum = 0;
  for(let i=0;i<probs.length;i++){ if(!isFinite(probs[i])||probs[i]<0) probs[i]=0; sum += probs[i]; }
  if(sum===0) return Math.floor(Math.random()*probs.length);
  let r = Math.random()*sum;
  for(let i=0;i<probs.length;i++){
    r -= probs[i];
    if(r <= 0) return i;
  }
  return probs.length-1;
}

// ----------------- Web search via CORS proxy (AllOrigins) -----------------
async function webSearchSnippet(query){
  try{
    const ddg = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const proxy = 'https://api.allorigins.win/get?url=' + encodeURIComponent(ddg);
    const r = await fetch(proxy);
    if(!r.ok) return null;
    const j = await r.json();
    const html = j.contents;
    // parse snippet: DuckDuckGo's markup uses <a class="result__a"> and <a class="result__snippet"> sometimes; we'll extract a few <a> text or paragraphs
    const doc = new DOMParser().parseFromString(html, 'text/html');
    // prefer result snippets
    let snippet = '';
    const snippetEl = doc.querySelector('.result__snippet');
    if(snippetEl) snippet = snippetEl.textContent.trim();
    if(!snippet){
      // fallback to first paragraph
      const p = doc.querySelector('p');
      if(p) snippet = p.textContent.trim();
    }
    // clean
    snippet = snippet.replace(/\s+/g, ' ').slice(0, 800);
    return snippet || null;
  }catch(e){ console.warn('webSearchSnippet failed', e); return null; }
}

// ----------------- Crawling pages for supervised training -----------------
/**
 * crawlPagesForQuery(query, maxPages)
 * - uses DuckDuckGo to get HTML results and then fetches the top result pages (via proxy)
 * - returns array [{url, text}, ...]
 */
export async function crawlPagesForQuery(query, maxPages=3){
  try{
    const ddg = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const proxySearch = 'https://api.allorigins.win/get?url=' + encodeURIComponent(ddg);
    const r = await fetch(proxySearch);
    if(!r.ok) return [];
    const j = await r.json();
    const doc = new DOMParser().parseFromString(j.contents, 'text/html');
    const links = Array.from(doc.querySelectorAll('a.result__a')).slice(0, maxPages).map(a => a.href);
    const pages = [];
    for(const url of links){
      try{
        const page = await fetch('https://api.allorigins.win/get?url=' + encodeURIComponent(url));
        if(!page.ok) continue;
        const pj = await page.json();
        const pageDoc = new DOMParser().parseFromString(pj.contents, 'text/html');
        // remove scripts
        pageDoc.querySelectorAll('script,style, noscript, iframe').forEach(n => n.remove());
        const text = pageDoc.body ? pageDoc.body.textContent.replace(/\s+/g,' ').slice(0, 20000) : '';
        pages.push({ url, text });
        // polite delay
        await new Promise(res => setTimeout(res, 600));
      }catch(e){ console.warn('page fetch failed', url, e); }
    }
    return pages;
  }catch(e){
    console.warn('crawlPagesForQuery failed', e);
    return [];
  }
}

// Expose crawling helper for the admin on window (used by index.js)
window.__crawlPagesForQuery = crawlPagesForQuery;

// ----------------- Supervised fine-tuning on text list (very light) -----------------
/**
 * supervisedTrainOnTexts(model, texts, epochs=1)
 * - breaks each text into sequences of tokens, constructs (context, next_token) pairs and runs model.fit
 * - keeps batch small to remain browser-friendly.
 */
export async function supervisedTrainOnTexts(model, texts, epochs=1){
  if(!texts || texts.length===0) return;
  const { vocab } = await loadVocab();
  const X = [];
  const Y = [];
  for(const t of texts){
    // split into sentences (rough)
    const sentences = (t || '').match(/[^.!?]+[.!?]?/g) || [];
    for(const s of sentences){
      const ids = textToIds(s, vocab); // length SEQ_LEN
      // pick target as next word after the sequence (if available), else 0
      const words = (s||'').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
      const target = words.length > 0 ? (vocab[words[0]] ?? 0) : 0; // simple target: first word
      X.push(ids);
      Y.push(target);
      if(X.length >= 128) break;
    }
    if(X.length >= 128) break;
  }
  if(X.length === 0) return;
  // train with small batch
  const xs = tf.tensor2d(X, [X.length, SEQ_LEN], 'int32');
  const ys = tf.tensor2d(Y, [Y.length, 1], 'int32');
  // compile temporarily for supervised fit
  model.compile({ optimizer: tf.train.adam(LR), loss: 'sparseCategoricalCrossentropy' });
  await model.fit(xs, ys, { epochs: epochs, batchSize: 16 });
  // dispose
  xs.dispose(); ys.dispose();
}

// ----------------- Reward (policy-gradient) update -----------------
/**
 * applyRewardUpdate(model, promptTokens, sampledTokens, reward)
 * - re-evaluates each sampling step under gradient tape and accumulates log-probs for sampled tokens.
 * - loss = - reward * sum(log_probs)
 * - performs a single optimizer step (low-variance but simple)
 */
export async function applyRewardUpdate(model, promptTokens, sampledTokens, rewardScalar=1.0){
  if(!model) throw new Error('Model missing');
  const { vocab } = await loadVocab();

  // optimizer.minimize takes a sync function that returns loss tf.Scalar
  optimizer.minimize(() => {
    let ctx = promptTokens.slice();
    let totalLogProb = tf.scalar(0);
    for(let t=0;t<sampledTokens.length;t++){
      const inputTensor = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
      const logits = model.apply(inputTensor); // [1, vocab]
      const logsoft = tf.logSoftmax(logits); // [1, vocab]
      // slice logprob of sampled token
      const idx = sampledTokens[t];
      // logsoft.slice([0, idx],[1,1]) -> shape [1,1], then squeeze
      const logp = logsoft.slice([0, idx],[1,1]).squeeze();
      totalLogProb = tf.add(totalLogProb, logp);
      // advance
      ctx = ctx.slice(1).concat([idx]);
      tf.dispose([inputTensor, logits, logsoft, logp]);
    }
    const loss = tf.mul(tf.scalar(-rewardScalar), totalLogProb);
    return loss;
  });

  await tf.nextFrame();
}

// ----------------- Upload model artifacts to Firebase Storage (using a custom save handler) -----------------
/**
 * saveModelArtifactsToFirebase(model, name, uploadFn)
 * - uploadFn should be a function (path:string, blob:Blob) => Promise<void>
 * - we use tf.io.withSaveHandler to capture artifacts and then call uploadFn for model.json and weights
 */
export async function saveModelArtifactsToFirebase(model, name='xylxx-model', uploadFn){
  if(!uploadFn) throw new Error('uploadFn required');
  const saveHandler = tf.io.withSaveHandler(async (artifacts) => {
    // artifacts: {modelTopology, weightSpecs, weightData (ArrayBuffer), ...}
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightSpecs: artifacts.weightSpecs,
      userDefinedMetadata: artifacts.userDefinedMetadata
    };
    const jsonBlob = new Blob([JSON.stringify(modelJson)], { type: 'application/json' });
    const weightsBlob = new Blob([artifacts.weightData], { type: 'application/octet-stream' });
    // upload both
    await uploadFn(`models/${name}/model.json`, jsonBlob);
    await uploadFn(`models/${name}/weights.bin`, weightsBlob);
    return { modelArtifactsInfo: { dateSaved: Date.now(), modelTopologyType: typeof artifacts.modelTopology, weightDataBytes: artifacts.weightData.byteLength } };
  });
  await model.save(saveHandler);
}

// convenience that uses firebase helper (uploadModelArtifactsToFirebase)
export async function uploadModelToFirebase(model, name='xylxx-model'){
  return await saveModelArtifactsToFirebase(model, name, uploadModelArtifactsToFirebase);
}

// export functions
export { saveModelToIndexedDB as saveModelToIndexedDB };
export { initModelFromIndexedDB as initModelFromIndexedDB };
export { supervisedTrainOnTexts as supervisedTrainOnTexts };
export { applyRewardUpdate as applyRewardUpdate };
export { generateWithWebBlend as generateWithWebBlend };
export { createModel as createModel };
