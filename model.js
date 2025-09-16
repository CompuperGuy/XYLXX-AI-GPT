// model.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

/*
CONFIG: tune these if you want more capacity (will increase memory & compute)
*/
const SEQ_LEN = 24;       // context window
const EMBED_DIM = 128;    // token embedding size
const N_HEADS = 4;        // multi-head attention
const N_LAYERS = 4;       // transformer blocks
const FF_DIM = 512;       // feed-forward inner dim
const MAX_GEN = 24;       // maximum tokens to generate
const LR = 3e-4;

// internal optimizer and helpers
const OPT = tf.train.adam(LR);

// load vocab.json into memory
export async function loadVocab(path='./vocab.json'){
  const r = await fetch(path);
  const j = await r.json();
  window.__XYLXX_VOCAB = j; // cache globally
  return j;
}

/* build positional embeddings as a tensor of shape [SEQ_LEN, EMBED_DIM] */
function positionalEncoding(seqLen, dim){
  const pe = new Float32Array(seqLen * dim);
  for(let pos=0; pos<seqLen; pos++){
    for(let i=0;i<dim;i++){
      const idx = pos*dim + i;
      const angle = pos / Math.pow(10000, (2*Math.floor(i/2))/dim);
      pe[idx] = (i%2===0) ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return tf.tensor2d(pe, [seqLen, dim]);
}

/* Multi-head attention implemented using tf.layers; returns a layer function */
function mhaLayer(embedDim, numHeads){
  const headDim = Math.floor(embedDim/numHeads);
  // We will implement multi-head by linear projections and manual reshape operations in model forward.
  // For training via tf.model.fit we construct a functional model where possible.
  // For our use-case (generation, reward updates), we'll use manual predict loops.
  return { embedDim, numHeads, headDim };
}

/* Create model object (not necessarily an official tf.Model) that contains predict step
   We create weight layers manually and attach forward/predict/generate/applyRewardUpdate methods.
*/
export async function makeModel(vocabSize = Object.keys(window.__XYLXX_VOCAB || await loadVocab()).length){
  // if vocab not loaded, make sure it's available
  const vocab = window.__XYLXX_VOCAB || await loadVocab();
  const invVocab = Object.fromEntries(Object.entries(vocab).map(([k,v])=>[v,k]));

  // Variables: embeddings, positional, then for each layer: q/k/v projections, out proj, layernorms, ff weights
  const embeddings = tf.variable(tf.randomNormal([vocabSize, EMBED_DIM], 0, 0.02), true, 'embeddings');
  const posEmb = tf.variable(positionalEncoding(SEQ_LEN, EMBED_DIM), false, 'posEmb'); // fixed but stored as var false

  // layer params
  const layers = [];
  for(let l=0;l<N_LAYERS;l++){
    // q,k,v linear: Wq, Wk, Wv -> [embed, embed]
    const Wq = tf.variable(tf.randomNormal([EMBED_DIM, EMBED_DIM],0,0.02));
    const Wk = tf.variable(tf.randomNormal([EMBED_DIM, EMBED_DIM],0,0.02));
    const Wv = tf.variable(tf.randomNormal([EMBED_DIM, EMBED_DIM],0,0.02));
    const Wo = tf.variable(tf.randomNormal([EMBED_DIM, EMBED_DIM],0,0.02));
    // layernorm scale & bias
    const ln1_gamma = tf.variable(tf.ones([EMBED_DIM]));
    const ln1_beta = tf.variable(tf.zeros([EMBED_DIM]));
    const ln2_gamma = tf.variable(tf.ones([EMBED_DIM]));
    const ln2_beta = tf.variable(tf.zeros([EMBED_DIM]));
    // feed-forward
    const W1 = tf.variable(tf.randomNormal([EMBED_DIM, FF_DIM],0,0.02));
    const b1 = tf.variable(tf.zeros([FF_DIM]));
    const W2 = tf.variable(tf.randomNormal([FF_DIM, EMBED_DIM],0,0.02));
    const b2 = tf.variable(tf.zeros([EMBED_DIM]));
    layers.push({ Wq,Wk,Wv,Wo,ln1_gamma,ln1_beta,ln2_gamma,ln2_beta,W1,b1,W2,b2 });
  }

  // final head
  const head = tf.variable(tf.randomNormal([EMBED_DIM, vocabSize],0,0.02));

  // Utility: layernorm
  function layerNorm(x, gamma, beta, eps=1e-5){
    const mean = tf.mean(x, -1, true);
    const variance = tf.mean(tf.square(tf.sub(x, mean)), -1, true);
    const normed = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, eps)));
    return tf.add(tf.mul(normed, gamma), beta);
  }

  // forward pass for a batch inputIds tensor shape [B, T]
  function forward(inputIds){
    return tf.tidy(()=>{
      // inputIds -> embeddings [B,T,EMB]
      const emb = tf.gather(embeddings, inputIds); // gather supports [B,T]
      // add positional embeddings (posEmb [T,EMB]) -> broadcast over batch
      const pos = posEmb.reshape([1, SEQ_LEN, EMBED_DIM]);
      let x = tf.add(emb, pos);
      // transformer layers
      for(let li=0; li<layers.length; li++){
        const L = layers[li];
        // self-attention (causal): compute q,k,v
        const q = tf.dot(x, L.Wq); // [B,T,EMB]
        const k = tf.dot(x, L.Wk);
        const v = tf.dot(x, L.Wv);
        // scaled dot-product attention with causal mask
        let scores = tf.matMul(q, k, false, true); // [B,T,T]
        scores = tf.div(scores, Math.sqrt(EMBED_DIM));
        // causal mask: set upper triangle to -inf
        const mask = tf.tile(tf.expandDims(tf.tensor2d(causalMask(T=SEQ_LEN)),0), [x.shape[0],1,1]); // [B,T,T]
        const MINF = -1e9;
        scores = tf.add(tf.mul(tf.cast(mask,'float32'), MINF), scores);
        const attn = tf.softmax(scores, -1);
        const context = tf.matMul(attn, v); // [B,T,EMB]
        const attnOut = tf.dot(context, L.Wo);
        // add & norm 1
        const x1 = layerNorm(tf.add(x, attnOut), L.ln1_gamma, L.ln1_beta);
        // feed-forward
        const ff = tf.add(tf.matMul(x1, L.W1), L.b1);
        const ffAct = tf.relu(ff);
        const ff2 = tf.add(tf.matMul(ffAct, L.W2), L.b2);
        // add & norm 2
        x = layerNorm(tf.add(x1, ff2), L.ln2_gamma, L.ln2_beta);
      }
      // take last token embedding for next-token logits
      const last = x.slice([0, SEQ_LEN-1, 0], [-1,1,-1]).reshape([x.shape[0], EMBED_DIM]);
      const logits = tf.matMul(last, head); // [B, V]
      return logits;
    });
  }

  // helper to create causal mask (boolean 2D array) where mask[i,j] is 0 when j<=i else 1
  function causalMask(T){
    const m = new Array(T*T).fill(0);
    for(let i=0;i<T;i++){
      for(let j=0;j<T;j++){
        if(j>i) m[i*T + j] = 1; // masked positions = 1 (we'll add MINF there)
      }
    }
    return new Float32Array(m);
  }

  // generation: autoregressive sampling using forward()
  async function generate(promptTokens, maxNew=MAX_GEN, temperature=1.0, topK=40){
    // promptTokens length <= SEQ_LEN (already padded/truncated)
    let ctx = promptTokens.slice(); // array of length SEQ_LEN
    const sampled = [];
    for(let step=0; step<maxNew; step++){
      const inp = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
      const logits = forward(inp); // [1, V]
      let logits1 = logits.div(tf.scalar(temperature));
      let probs = tf.softmax(logits1).dataSync(); // read as Float32Array
      // top-k sampling
      const nextIdx = sampleTopK(probs, topK);
      sampled.push(nextIdx);
      // advance ctx
      ctx = ctx.slice(1).concat([nextIdx]);
      tf.dispose([inp, logits, logits1]);
    }
    return sampled;
  }

  function sampleTopK(probs, k){
    // returns index sampled from top-k probabilities
    const arr = Array.from(probs).map((p,i)=>({i,p}));
    arr.sort((a,b)=>b.p - a.p);
    const top = arr.slice(0,k);
    const s = top.reduce((a,b)=>a+b.p,0);
    let r = Math.random()*s;
    for(const t of top){
      r -= t.p;
      if(r<=0) return t.i;
    }
    return top[0].i;
  }

  // REINFORCE-style update: compute differentiable log-probs and apply gradient step
  async function applyRewardUpdate(promptTokens, sampledTokens, rewardValue=1.0){
    // optimizer.minimize requires a function returning scalar loss
    await OPT.minimize(()=>{
      let ctx = promptTokens.slice();
      let acc = tf.scalar(0);
      for(let t=0;t<sampledTokens.length;t++){
        const inp = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
        const logits = forward(inp); // [1,V]
        const logp = tf.logSoftmax(logits).squeeze(); // [V]
        const picked = logp.gather(tf.tensor1d([sampledTokens[t]], 'int32')).squeeze(); // scalar
        acc = tf.add(acc, picked);
        ctx = ctx.slice(1).concat([sampledTokens[t]]);
        tf.dispose([inp, logits, logp, picked]);
      }
      const loss = tf.mul(tf.scalar(-rewardValue), acc); // -R * sum logp
      return loss;
    }, true);
    await tf.nextFrame();
  }

  // supervised fine-tune on list of texts (light)
  async function supervisedTrainOnTexts(texts, vocab, epochs=1){
    // create training pairs: X = context (SEQ_LEN tokens), Y = next token id
    const X = []; const Y = [];
    for(const t of texts){
      const words = (t||'').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
      const ids = words.map(w => vocab[w] ?? 0);
      for(let i=0;i+SEQ_LEN < ids.length;i++){
        X.push(ids.slice(i, i+SEQ_LEN));
        Y.push(ids[i+SEQ_LEN]);
        if(X.length >= 512) break;
      }
      if(X.length >= 512) break;
    }
    if(X.length === 0) return;
    // convert to tensors and train with simple dense head scheme: we can approximate by training head
    const xs = tf.tensor2d(X, [X.length, SEQ_LEN], 'int32');
    const ys = tf.tensor1d(Y, 'int32');
    // We'll create a small compile-and-fit wrapper: compute logits via forward and minimize cross-entropy
    // Use custom training loop for stability
    const BATCH = 32;
    for(let e=0;e<epochs;e++){
      for(let i=0;i<xs.shape[0]; i+=BATCH){
        const xb = xs.slice([i,0],[Math.min(BATCH, xs.shape[0]-i), SEQ_LEN]);
        const yb = ys.slice([i],[Math.min(BATCH, xs.shape[0]-i)]);
        await OPT.minimize(()=>{
          const logits = forward(xb); // [B, V]
          const loss = tf.losses.sparseSoftmaxCrossEntropy(yb, logits);
          tf.dispose(logits);
          return loss;
        }, true);
        tf.dispose([xb, yb]);
        await tf.nextFrame();
      }
    }
    xs.dispose(); ys.dispose();
  }

  // utilities exposed by model wrapper
  const modelWrapper = {
    vocab, invVocab,
    forward, generate, applyRewardUpdate,
    supervisedTrainOnTexts,
    // wrapper helper used by app_boot: call generate then map ids->words
    async generateTextAndRecord(promptText, topK=40, temperature=1.0){
      const vocabMap = vocab;
      // build prompt tokens padded/truncated to SEQ_LEN (left pad with zeros)
      const words = (promptText||'').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
      const ids = words.map(w => vocabMap[w] ?? 0);
      const pad = Math.max(0, SEQ_LEN - ids.length);
      const promptTokens = new Array(pad).fill(0).concat(ids.slice(-SEQ_LEN));
      // actual generate
      const sampled = await generate(promptTokens, MAX_GEN, temperature, topK);
      // convert sampled tokens to words
      const text = sampled.map(i => invVocab[i] ?? '…').join(' ');
      const turnId = 't_' + Date.now().toString(36) + '_' + Math.floor(Math.random()*9999).toString(36);
      return { replyText: text, promptTokens, sampledTokens: sampled, turnId };
    },
    // persistence helpers
    async saveIndexedDB(name='xylxx-model'){
      // pack vars into a tfjs-format model using save handler
      // For speed/size we only persist the important weights (embeddings and head + layers)
      // Create a minimal tf.Model that maps input tokens->logits using current variables then save
      // Build functional model for saving
      const input = tf.input({shape:[SEQ_LEN], dtype:'int32'});
      // simple gather via embedding matrix using tf.layers.embedding isn't trivial to connect to existing variable,
      // but tf.layers.embedding can be created with weights initialised from embeddings array
      const embArr = embeddings.arraySync();
      const embedLayer = tf.layers.embedding({inputDim:embArr.length, outputDim:EMBED_DIM, weights:[tf.tensor(embArr)], trainable:true});
      const embOut = embedLayer.apply(input);
      // For saving we won't replicate transformer; instead save just embeddings and a small head to keep state.
      const pool = tf.layers.flatten().apply(embOut);
      const out = tf.layers.dense({units:Object.keys(vocab).length, activation:null}).apply(pool);
      const saveModel = tf.model({inputs:input, outputs:out});
      await saveModel.save(`indexeddb://${name}`);
      saveModel.dispose();
      embedLayer.dispose();
    }
  };

  // attach convenience method used by app_boot to call reward update on model object
  modelWrapper.applyRewardUpdate = applyRewardUpdate;

  return modelWrapper;
}

/* helper: try load model from IndexedDB named 'xylxx-init' (if created earlier)
   For simplicity the save/load mapping is handled in app flow elsewhere.
*/
export async function loadModelFromIndexedDB(){
  try{
    const models = await tf.io.listModels();
    const key = 'indexeddb://xylxx-init';
    if(models[key]){
      const load = await tf.loadLayersModel(key);
      // That saved model will not have our custom variables; instead we fall back to makeModel
      // To keep flow simple, return null to trigger fresh makeModel
      return null;
    }
    return null;
  }catch(e){ console.warn('loadModelFromIndexedDB', e); return null; }
}

/* generateResponseAndRecord: used by app_boot.js
   returns: { reply: string, turnId, promptTokens, sampledTokens }
*/
export async function generateResponseAndRecord(modelWrapper, promptText, vocab, invVocab){
  const out = await modelWrapper.generateTextAndRecord(promptText, 40, 1.0);
  // optionally perform web snippet fetch and blend (we call a helper)
  const snippet = await webSnippetForQuery(promptText);
  let reply = out.replyText;
  if(snippet && snippet.length>20){
    reply = `${reply}\n\n(From web): ${snippet}`;
  }
  // kick background crawl->supervised fine-tune
  if(snippet){
    (async ()=>{
      try{
        const pages = await crawlPagesForQuery(promptText, 2);
        if(pages.length) await modelWrapper.supervisedTrainOnTexts(pages.map(p=>p.text), vocab, 1);
      }catch(e){ console.warn('bg crawl/train failed', e); }
    })();
  }
  return { reply, turnId: out.turnId, promptTokens: out.promptTokens, sampledTokens: out.sampledTokens };
}

/* web helper: use DuckDuckGo via allorigins proxy to fetch snippet */
async function webSnippetForQuery(q){
  try{
    const ddg = `https://duckduckgo.com/html/?q=${encodeURIComponent(q)}`;
    const proxy = 'https://api.allorigins.win/get?url=' + encodeURIComponent(ddg);
    const r = await fetch(proxy);
    if(!r.ok) return null;
    const j = await r.json();
    const doc = new DOMParser().parseFromString(j.contents, 'text/html');
    const s = doc.querySelector('.result__snippet') || doc.querySelector('p');
    return s ? s.textContent.trim().replace(/\s+/g,' ').slice(0,800) : null;
  }catch(e){ return null; }
}

/* crawlPagesForQuery: fetch top result links and retrieve page text (via proxy) */
async function crawlPagesForQuery(query, maxPages=2){
  try{
    const ddg = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const proxySearch = 'https://api.allorigins.win/get?url=' + encodeURIComponent(ddg);
    const r = await fetch(proxySearch);
    if(!r.ok) return [];
    const j = await r.json();
    const doc = new DOMParser().parseFromString(j.contents, 'text/html');
    const anchors = Array.from(doc.querySelectorAll('a.result__a')).slice(0, maxPages);
    const pages = [];
    for(const a of anchors){
      let url = a.href;
      // fetch page via allorigins
      try{
        const pr = await fetch('https://api.allorigins.win/get?url=' + encodeURIComponent(url));
        if(!pr.ok) continue;
        const pj = await pr.json();
        const pdoc = new DOMParser().parseFromString(pj.contents, 'text/html');
        pdoc.querySelectorAll('script,style,iframe,noscript').forEach(n=>n.remove());
        const text = (pdoc.body ? pdoc.body.textContent : '').replace(/\s+/g,' ').slice(0,20000);
        pages.push({url, text});
        await new Promise(r=>setTimeout(r,600));
      }catch(e){ continue; }
    }
    return pages;
  }catch(e){ return []; }
}
