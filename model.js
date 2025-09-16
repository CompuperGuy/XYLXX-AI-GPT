// model.js
// Advanced browser-side Transformer-like model with generation, reward updates, supervised fine-tuning, and persistence.
// Requires TensorFlow.js loaded in the page (e.g. <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>)
// Exports: initModel, generateAndRecord, applyRewardUpdate, supervisedTrainOnTexts, saveModelIndexedDB, loadModelIndexedDB

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

/* =========================
   CONFIGURATION
   ========================= */
const CONFIG = {
  SEQ_LEN: 24,           // context window length (tokens)
  EMBED_DIM: 160,        // token embedding dimension
  N_LAYERS: 6,           // number of transformer blocks
  N_HEADS: 8,            // number of attention heads
  FF_DIM: 512,           // feed-forward inner dimension
  MAX_GEN: 24,           // max tokens to generate
  TOP_K: 40,             // top-k sampling
  TEMPERATURE: 1.0,      // sampling temperature
  LEARNING_RATE: 2e-4,   // optimizer learning rate
  SAVE_KEY: 'indexeddb://xylxx-transformer'
};

const MINF = -1e9; // used for masking

/* =========================
   GLOBALS / CACHES
   ========================= */
let GLOBAL = {
  modelWrapper: null,   // returned model wrapper object
  vocab: null,          // word->index map
  invVocab: null,       // index->word map
  optimizer: null
};

/* =========================
   UTILITIES
   ========================= */

/** load JSON vocab from given path (expects {"word": idx, ...}) */
export async function loadVocab(path = './vocab.json') {
  const r = await fetch(path);
  if (!r.ok) throw new Error('Could not load vocab.json');
  const j = await r.json();
  GLOBAL.vocab = j;
  GLOBAL.invVocab = Object.fromEntries(Object.entries(j).map(([w, i]) => [i, w]));
  return GLOBAL.vocab;
}

/** simple tokenizer: split on word boundaries, use vocab (unknown => 0) */
export function textToIds(text, padLeft = true) {
  if (!GLOBAL.vocab) throw new Error('Vocab not loaded');
  const words = (text || '').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
  const ids = words.map(w => (w in GLOBAL.vocab ? GLOBAL.vocab[w] : 0));
  // pad or truncate to SEQ_LEN
  const L = CONFIG.SEQ_LEN;
  if (ids.length >= L) return ids.slice(-L);
  if (padLeft) return new Array(L - ids.length).fill(0).concat(ids);
  return ids.concat(new Array(L - ids.length).fill(0));
}

/** ids to text using invVocab */
export function idsToText(ids) {
  if (!GLOBAL.invVocab) throw new Error('invVocab missing');
  return ids.map(i => GLOBAL.invVocab[i] ?? '…').join(' ');
}

/* =========================
   Linear algebra helpers
   ========================= */

/** Create a causal mask (SEQ_LEN x SEQ_LEN) with 0 allowed, 1 masked (upper triangle) */
function makeCausalMask(L = CONFIG.SEQ_LEN) {
  const a = new Float32Array(L * L);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      a[i * L + j] = j > i ? 1.0 : 0.0;
    }
  }
  // return as tensor of shape [1, L, L] to broadcast over batch
  return tf.tensor(a, [1, L, L]);
}

/* =========================
   MODEL CONSTRUCTION (weights as tf.Variables)
   The model is implemented as a custom wrapper, not necessarily a single tf.Model,
   because we need fine control during generation and reward update loops.
   ========================= */

function randn(shape, std = 0.02) {
  return tf.randomNormal(shape, 0, std);
}

function glorot(shape) {
  const fanIn = shape[0];
  const fanOut = shape[1] ?? shape[shape.length - 1];
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return tf.randomUniform(shape, -limit, limit);
}

/** Build a new transformer-like model wrapper */
async function buildTransformer(vocabSize) {
  // initialize optimizer
  GLOBAL.optimizer = tf.train.adam(CONFIG.LEARNING_RATE);

  // embedding matrix (vocabSize x EMBED_DIM)
  const Wemb = tf.variable(randn([vocabSize, CONFIG.EMBED_DIM]), true, 'Wemb');

  // positional embeddings (SEQ_LEN x EMBED_DIM) - constant
  const posArray = new Float32Array(CONFIG.SEQ_LEN * CONFIG.EMBED_DIM);
  for (let pos = 0; pos < CONFIG.SEQ_LEN; pos++) {
    for (let i = 0; i < CONFIG.EMBED_DIM; i++) {
      const idx = pos * CONFIG.EMBED_DIM + i;
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / CONFIG.EMBED_DIM);
      posArray[idx] = (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }
  }
  const Wpos = tf.tensor(posArray, [CONFIG.SEQ_LEN, CONFIG.EMBED_DIM]);

  // per-layer weights
  const layers = [];
  for (let L = 0; L < CONFIG.N_LAYERS; L++) {
    // query/key/value projection weights: [EMBED_DIM, EMBED_DIM]
    const Wq = tf.variable(glorot([CONFIG.EMBED_DIM, CONFIG.EMBED_DIM]), true, `L${L}_Wq`);
    const Wk = tf.variable(glorot([CONFIG.EMBED_DIM, CONFIG.EMBED_DIM]), true, `L${L}_Wk`);
    const Wv = tf.variable(glorot([CONFIG.EMBED_DIM, CONFIG.EMBED_DIM]), true, `L${L}_Wv`);
    const Wo = tf.variable(glorot([CONFIG.EMBED_DIM, CONFIG.EMBED_DIM]), true, `L${L}_Wo`);

    // layernorm params (pre-attn)
    const ln1_gamma = tf.variable(tf.ones([CONFIG.EMBED_DIM]), true, `L${L}_ln1_g`);
    const ln1_beta = tf.variable(tf.zeros([CONFIG.EMBED_DIM]), true, `L${L}_ln1_b`);
    // layernorm params (post-ff)
    const ln2_gamma = tf.variable(tf.ones([CONFIG.EMBED_DIM]), true, `L${L}_ln2_g`);
    const ln2_beta = tf.variable(tf.zeros([CONFIG.EMBED_DIM]), true, `L${L}_ln2_b`);

    // feed-forward weights
    const W1 = tf.variable(glorot([CONFIG.EMBED_DIM, CONFIG.FF_DIM]), true, `L${L}_W1`);
    const b1 = tf.variable(tf.zeros([CONFIG.FF_DIM]), true, `L${L}_b1`);
    const W2 = tf.variable(glorot([CONFIG.FF_DIM, CONFIG.EMBED_DIM]), true, `L${L}_W2`);
    const b2 = tf.variable(tf.zeros([CONFIG.EMBED_DIM]), true, `L${L}_b2`);

    layers.push({ Wq, Wk, Wv, Wo, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, W1, b1, W2, b2 });
  }

  // final lm head
  const head = tf.variable(glorot([CONFIG.EMBED_DIM, vocabSize]), true, 'head');

  // causal mask constant
  const causalMask = makeCausalMask(CONFIG.SEQ_LEN); // shape [1,L,L]

  /* ---------- helper ops ---------- */

  function layerNorm(x, gamma, beta, eps = 1e-5) {
    // x shape [..., EMBED_DIM]
    const mean = tf.mean(x, -1, true);
    const variance = tf.mean(tf.square(tf.sub(x, mean)), -1, true);
    const normed = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, eps)));
    return tf.add(tf.mul(normed, gamma), beta);
  }

  /** split heads: from [B, T, EMBED] to [B, numHeads, T, headDim] */
  function splitHeads(x) {
    const [B, T, E] = x.shape;
    const headDim = Math.floor(E / CONFIG.N_HEADS);
    return x.reshape([B, T, CONFIG.N_HEADS, headDim]).transpose([0, 2, 1, 3]); // [B, H, T, D]
  }

  /** combine heads: [B, H, T, D] -> [B, T, H*D] */
  function combineHeads(x) {
    const t = x.transpose([0, 2, 1, 3]); // [B, T, H, D]
    const shape = t.shape;
    return t.reshape([shape[0], shape[1], shape[2] * shape[3]]); // [B, T, E]
  }

  /* ---------- forward pass ---------- */
  function forward(inputIds) {
    // inputIds: tf.Tensor int32 shape [B, SEQ_LEN]
    return tf.tidy(() => {
      const B = inputIds.shape[0];

      // embeddings: gather rows
      const emb = tf.gather(Wemb, inputIds); // shape [B, T, E]
      // add positional embeddings (Wpos: [T,E]) -> broadcast
      const pos = Wpos.reshape([1, CONFIG.SEQ_LEN, CONFIG.EMBED_DIM]);
      let x = tf.add(emb, pos); // [B, T, E]

      // transformer layers
      for (let li = 0; li < layers.length; li++) {
        const L = layers[li];

        // Layer norm pre-attn
        const x_ln1 = layerNorm(x, L.ln1_gamma, L.ln1_beta); // [B,T,E]

        // q,k,v projections
        const q = tf.dot(x_ln1, L.Wq); // [B,T,E]
        const k = tf.dot(x_ln1, L.Wk);
        const v = tf.dot(x_ln1, L.Wv);

        // split heads -> [B,H,T,D]
        const qh = splitHeads(q);
        const kh = splitHeads(k);
        const vh = splitHeads(v);

        // scaled dot-product attention: scores [B, H, T, T]
        let scores = tf.matMul(qh, kh, false, true); // [B,H,T,T]
        scores = tf.div(scores, Math.sqrt(CONFIG.EMBED_DIM / CONFIG.N_HEADS));

        // apply causal mask: mask shape [1, T, T], broadcast to [B,H,T,T]
        const maskAdd = causalMask.reshape([1, 1, CONFIG.SEQ_LEN, CONFIG.SEQ_LEN])
          .mul(tf.scalar(MINF)); // masked entries get -inf
        scores = tf.add(scores, maskAdd);

        // softmax
        const weights = tf.softmax(scores, -1); // [B,H,T,T]

        // attention output = weights @ v
        const contextHeads = tf.matMul(weights, vh); // [B,H,T,D]

        // combine heads
        const context = combineHeads(contextHeads); // [B, T, E]

        // output projection
        const attnOut = tf.dot(context, L.Wo); // [B,T,E]

        // residual
        const x2 = tf.add(x, attnOut);

        // layer norm 2 pre-ff? We'll use x2 normalized as input to FF
        const x_ln2 = layerNorm(x2, L.ln2_gamma, L.ln2_beta);

        // feed-forward
        const ff1 = tf.add(tf.dot(x_ln2, L.W1), L.b1); // [B,T,FF]
        const ffAct = tf.relu(ff1);
        const ff2 = tf.add(tf.dot(ffAct, L.W2), L.b2); // [B,T,E]

        // residual
        x = tf.add(x2, ff2);
      }

      // prepare logits from last token only (auto-regressive)
      // take last time index
      const last = x.slice([0, CONFIG.SEQ_LEN - 1, 0], [-1, 1, -1]).reshape([B, CONFIG.EMBED_DIM]); // [B, E]
      const logits = tf.dot(last, head); // [B, V]
      return logits;
    });
  }

  /* generation helper (autoregressive sampling) */
  async function generate(promptTokens, options = {}) {
    const topK = options.topK ?? CONFIG.TOP_K;
    const temperature = options.temperature ?? CONFIG.TEMPERATURE;
    const maxNew = options.maxNew ?? CONFIG.MAX_GEN;

    // promptTokens: array length SEQ_LEN
    let ctx = promptTokens.slice(); // keep as JS array
    const sampled = [];
    for (let t = 0; t < maxNew; t++) {
      // create input tensor
      const inp = tf.tensor2d([ctx], [1, CONFIG.SEQ_LEN], 'int32');
      const logits = forward(inp); // [1, V]
      // apply temperature and get probabilities
      const scaled = tf.div(logits, tf.scalar(Math.max(1e-8, temperature)));
      const probs = tf.softmax(scaled).dataSync(); // Float32Array of length V
      // sample from top-k
      const next = sampleTopK(probs, topK);
      sampled.push(next);
      // advance ctx
      ctx = ctx.slice(1).concat([next]);
      tf.dispose([inp, logits, scaled]);
      // small yield to event loop occasionally
      if (t % 6 === 0) await tf.nextFrame();
    }
    return sampled;
  }

  /** top-k sampling from probability array */
  function sampleTopK(probs, k) {
    const arr = Array.from(probs).map((p, i) => ({ p, i }));
    arr.sort((a, b) => b.p - a.p);
    const top = arr.slice(0, Math.min(k, arr.length));
    const s = top.reduce((acc, x) => acc + x.p, 0);
    let r = Math.random() * s;
    for (const item of top) {
      r -= item.p;
      if (r <= 0) return item.i;
    }
    return top[0].i;
  }

  /* reward update (REINFORCE-like)
     - given promptTokens (array length SEQ_LEN) and sampledTokens (array),
     - compute differentiable log-probs under current params and perform one gradient step
     - loss = - reward * sum(log_probs)
  */
  async function applyRewardUpdate(promptTokens, sampledTokens, rewardScalar = 1.0) {
    if (!Array.isArray(promptTokens) || !Array.isArray(sampledTokens)) {
      throw new Error('promptTokens and sampledTokens must be arrays');
    }
    // Use optimizer.minimize with a function returning a scalar tf.Scalar
    await GLOBAL.optimizer.minimize(() => {
      // We'll accumulate log probs
      let ctx = promptTokens.slice();
      let acc = tf.scalar(0);
      for (let t = 0; t < sampledTokens.length; t++) {
        const inp = tf.tensor2d([ctx], [1, CONFIG.SEQ_LEN], 'int32');
        const logits = forward(inp); // [1, V]
        const logp = tf.logSoftmax(logits).squeeze(); // [V]
        const idx = sampledTokens[t];
        // gather logp at idx
        const picked = logp.slice([idx], [1]).squeeze(); // scalar
        acc = tf.add(acc, picked);
        // advance
        ctx = ctx.slice(1).concat([idx]);
        // free interim (but keep acc)
        tf.dispose([inp, logits, logp, picked]);
      }
      // loss = -reward * acc
      const loss = tf.mul(tf.scalar(-rewardScalar), acc);
      return loss;
    }, true);
    await tf.nextFrame();
  }

  /* Supervised fine-tuning on texts (lightweight)
     texts: array of plain strings. We construct (context->next_token) examples by sliding window.
     This is a small batch trainer and will only use a subset to remain browser-friendly.
  */
  async function supervisedTrainOnTexts(texts = [], epochs = 1, maxExamples = 512) {
    if (!Array.isArray(texts) || texts.length === 0) return 0;
    if (!GLOBAL.vocab) throw new Error('vocab not loaded');
    // build dataset
    const examplesX = [];
    const examplesY = [];
    for (const t of texts) {
      const words = (t || '').toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
      const ids = words.map(w => (w in GLOBAL.vocab ? GLOBAL.vocab[w] : 0));
      for (let i = 0; i + CONFIG.SEQ_LEN < ids.length; i++) {
        examplesX.push(ids.slice(i, i + CONFIG.SEQ_LEN));
        examplesY.push(ids[i + CONFIG.SEQ_LEN]); // next token
        if (examplesX.length >= maxExamples) break;
      }
      if (examplesX.length >= maxExamples) break;
    }
    if (examplesX.length === 0) return 0;

    const X = tf.tensor2d(examplesX, [examplesX.length, CONFIG.SEQ_LEN], 'int32'); // [N, T]
    const Y = tf.tensor1d(examplesY, 'int32'); // [N]

    // custom training loop: minimize cross-entropy of logits vs Y
    for (let e = 0; e < epochs; e++) {
      const batchSize = 32;
      for (let i = 0; i < X.shape[0]; i += batchSize) {
        const end = Math.min(i + batchSize, X.shape[0]);
        const xb = X.slice([i, 0], [end - i, CONFIG.SEQ_LEN]);
        const yb = Y.slice([i], [end - i]);
        await GLOBAL.optimizer.minimize(() => {
          const logits = forward(xb); // [B, V]
          const loss = tf.losses.sparseSoftmaxCrossEntropy(yb, logits);
          tf.dispose(logits);
          return loss;
        }, true);
        tf.dispose([xb, yb]);
        await tf.nextFrame();
      }
    }
    tf.dispose([X, Y]);
    return examplesX.length;
  }

  /* persistence: save a small surrogate model to IndexedDB
     Because our variables are custom tf.Variables, easiest approach is to build a tiny
     tf.Model that we initialize from current weights for portable saving.
     We will persist embeddings and head via a small functional model: tokens -> logits
  */
  async function saveModelIndexedDB(name = CONFIG.SAVE_KEY) {
    // Create a small tf.Model that uses current embedding and head weights
    // Use tf.layers.embedding with weights initialized to Wemb
    const vocabSize = GLOBAL.vocab ? Object.keys(GLOBAL.vocab).length : head.shape[1];
    // gather embedding as array
    const embArr = await Wemb.array();
    // build model
    const input = tf.input({ shape: [CONFIG.SEQ_LEN], dtype: 'int32' });
    // layers.embedding with weights
    const embedLayer = tf.layers.embedding({ inputDim: embArr.length, outputDim: CONFIG.EMBED_DIM, weights: [tf.tensor(embArr)], trainable: true });
    const embOut = embedLayer.apply(input); // [B,T,E]
    // simple pooling - take last token (mimic our forward)
    const last = tf.layers.lambda(x => tf.slice(x, [0, CONFIG.SEQ_LEN - 1, 0], [-1, 1, -1])).apply(embOut);
    const flat = tf.layers.flatten().apply(last); // [B,E]
    const denseOut = tf.layers.dense({ units: head.shape[1], useBias: false, weights: [head] }).apply(flat); // [B,V]
    const saveModel = tf.model({ inputs: input, outputs: denseOut });
    // save
    await saveModel.save(name);
    // cleanup
    saveModel.dispose();
    embedLayer.dispose();
  }

  async function loadModelIndexedDB(name = CONFIG.SAVE_KEY) {
    try {
      const models = await tf.io.listModels();
      if (models[name]) {
        // if exists, load the lightweight model; but we cannot easily restore our custom variables automatically.
        // We return the loaded tf.Model as a fallback; to keep consistent logic, we signal null and let the app create new wrapper.
        const loaded = await tf.loadLayersModel(name);
        return loaded;
      }
      return null;
    } catch (e) {
      console.warn('loadModelIndexedDB error', e);
      return null;
    }
  }

  /* wrapper object exposing functions needed by UI */
  const wrapper = {
    forward,
    generate: generate,
    applyRewardUpdate,
    supervisedTrainOnTexts,
    saveModelIndexedDB,
    loadModelIndexedDB,
    // convenience: generate text and return useful records for reward
    async generateTextAndRecord(promptText, opts = {}) {
      if (!GLOBAL.vocab) throw new Error('vocab not loaded');
      // prompt -> tokens (pad/truncate)
      const promptTokens = textToIds(promptText);
      const sampled = await generate(promptTokens, opts);
      const text = sampled.map(i => GLOBAL.invVocab[i] ?? '…').join(' ');
      const turnId = 't_' + Date.now().toString(36) + '_' + Math.floor(Math.random() * 10000).toString(36);
      return {
        replyText: text,
        promptTokens,
        sampledTokens: sampled,
        turnId
      };
    }
  };

  return wrapper;
}


export async function initModel(options = {}) {
  const vocabPath = options.vocabPath || './vocab.json';
  if (!GLOBAL.vocab) await loadVocab(vocabPath);
  // build transformer wrapper
  const vocabSize = Object.keys(GLOBAL.vocab).length;
  GLOBAL.modelWrapper = await buildTransformer(vocabSize);
  return GLOBAL.modelWrapper;
}

/** generateAndRecord(promptText, opts)
 * returns { replyText, promptTokens, sampledTokens, turnId }
 */
export async function generateAndRecord(promptText, opts = { temperature: CONFIG.TEMPERATURE, topK: CONFIG.TOP_K, maxNew: CONFIG.MAX_GEN }) {
  if (!GLOBAL.modelWrapper) throw new Error('Model not initialized - call initModel() first');
  return await GLOBAL.modelWrapper.generateTextAndRecord(promptText, opts);
}

/** applyRewardUpdate(promptTokens, sampledTokens, reward)
 * wrapper to call model's reward update
 */
export async function applyRewardUpdate(promptTokens, sampledTokens, reward = 1.0) {
  if (!GLOBAL.modelWrapper) throw new Error('Model not initialized - call initModel() first');
  return await GLOBAL.modelWrapper.applyRewardUpdate(promptTokens, sampledTokens, reward);
}

/** supervisedTrainOnTexts(textArray, epochs=1)
 * returns number of examples used
 */
export async function supervisedTrainOnTexts(textArray = [], epochs = 1) {
  if (!GLOBAL.modelWrapper) throw new Error('Model not initialized - call initModel() first');
  return await GLOBAL.modelWrapper.supervisedTrainOnTexts(textArray, epochs);
}

/** saveModelIndexedDB(name) - save a lightweight surrogate to IndexedDB */
export async function saveModelIndexedDB(name = CONFIG.SAVE_KEY) {
  if (!GLOBAL.modelWrapper) throw new Error('Model not initialized - call initModel() first');
  return await GLOBAL.modelWrapper.saveModelIndexedDB(name);
}

/** loadModelIndexedDB(name) - attempt to load saved model (fallback) */
export async function loadModelIndexedDB(name = CONFIG.SAVE_KEY) {
  // this attempts to load a previously saved lightweight model; returns tf.Model or null
  try {
    const models = await tf.io.listModels();
    if (models[name]) {
      const m = await tf.loadLayersModel(name);
      return m;
    }
    return null;
  } catch (e) {
    console.warn('loadModelIndexedDB failed', e);
    return null;
  }
}

/* =========================
   SMALL DEBUG HELPERS (optional)
   ========================= */

export function setConfig(newCfg) {
  Object.assign(CONFIG, newCfg);
}

export function getConfig() {
  return { ...CONFIG };
}
