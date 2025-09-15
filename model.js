// model.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

/*
  Tiny policy network for token generation.
  - Fixed sequence length (seqLen) for prompt/context (we pad/truncate).
  - Embedding -> simple self-attention block (single head, small dim) -> dense -> vocab softmax.
  - generateWithRecording: samples tokens autoregressively, stores sampled token indices.
  - applyRewardUpdate: re-runs forward under gradients, gathers log-probs of those sampled tokens and
    performs a small policy-gradient step: loss = -reward * sum(log_probs).
*/

const SEQ_LEN = 12;        // context window
const MAX_GEN = 10;        // generate up to 10 tokens
const EMBED_DIM = 64;
const ATT_DIM = 64;
const HIDDEN = 128;
const LEARNING_RATE = 5e-4;

let optimizer = tf.train.adam(LEARNING_RATE);

/** load vocab json */
export async function loadVocab(url){
  const r = await fetch(url);
  const j = await r.json();
  return j; // { word: idx, ... }
}

/** Create a small functional model */
export function createModel(vocabSize){
  // inputs: int32 tensor shape [batch=1, SEQ_LEN]
  const input = tf.input({shape:[SEQ_LEN], dtype:'int32', name:'input_tokens'});

  // embedding
  const embedLayer = tf.layers.embedding({ inputDim: vocabSize, outputDim: EMBED_DIM, embeddingsInitializer:'glorotUniform' });
  const embeddings = embedLayer.apply(input); // shape [batch, SEQ_LEN, EMBED_DIM]

  // simple self-attention (single head)
  const q = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embeddings);
  const k = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embeddings);
  const v = tf.layers.dense({units:ATT_DIM, useBias:false}).apply(embeddings);

  // scaled dot product: scores = q dot k^T
  const scores = tf.layers.dot({axes:[2,2]}).apply([q,k]); // [batch, SEQ_LEN, SEQ_LEN]
  const scaled = tf.layers.lambda(x => tf.mul(x, 1/Math.sqrt(ATT_DIM))).apply(scores);
  const weights = tf.layers.activation({activation:'softmax'}).apply(scaled);
  const context = tf.layers.dot({axes:[2,1]}).apply([weights, v]); // [batch, SEQ_LEN, ATT_DIM]

  // flatten and a couple dense layers
  const flat = tf.layers.flatten().apply(context);
  const d1 = tf.layers.dense({units:HIDDEN, activation:'relu'}).apply(flat);
  const logits = tf.layers.dense({units:vocabSize}).apply(d1); // [batch, vocabSize]
  // Note: model returns logits for the *next* token prediction given last SEQ_LEN tokens.

  const model = tf.model({inputs: input, outputs: logits});
  return model;
}

/** Helper: convert words -> token ids with vocab, pad/truncate to SEQ_LEN */
export function textToTokenIds(text, vocab){
  const tokens = text.toLowerCase().match(/\b\w+'\w+|\b\w+\b/g) || [];
  const ids = tokens.map(w => (w in vocab) ? vocab[w] : 0);
  // pad left with zeros if shorter (simple)
  const out = new Array(Math.max(0, SEQ_LEN - ids.length)).fill(0).concat(ids.slice(-SEQ_LEN));
  return out.slice(-SEQ_LEN);
}

/** Helper: token ids -> text using inverse vocab map */
export function idsToText(ids, invVocab){
  const words = ids.map(i => invVocab[i] ?? '…');
  return words.join(' ');
}

/** Generate autoregressively, record sampled tokens and return turnId */
export async function generateWithRecording(model, vocab, promptText, invVocab, temperature=1.0){
  // prompt -> token ids (length SEQ_LEN)
  const promptIds = textToTokenIds(promptText, vocab);
  // We'll sample up to MAX_GEN tokens one-by-one. For each step, we create input array of last SEQ_LEN tokens.
  const sampled = [];
  // for stability convert to numbers
  let ctx = promptIds.slice();

  // sample loop (synchronous tf.tidy around predict)
  for(let step=0; step<MAX_GEN; step++){
    const inputArr = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
    const logits = tf.tidy(()=> model.predict(inputArr)); // [1, vocabSize]
    // apply temperature
    let probs = tf.tidy(()=> tf.softmax(tf.div(logits, tf.scalar(Math.max(1e-8, temperature)))));
    const probsData = await probs.data();
    // sample index according to probs
    const idx = sampleFromProbs(probsData);
    sampled.push(idx);
    // advance context: drop first, push idx
    ctx = ctx.slice(1).concat([idx]);
    tf.dispose([inputArr, logits, probs]);
  }

  // assemble reply text (map sampled ids to words)
  const replyText = idsToText(sampled, invVocab);

  // create a lightweight turnId
  const turnId = 't_' + Date.now().toString(36) + '_' + Math.floor(Math.random()*9999).toString(36);

  // return all pieces
  return { replyText, promptTokens: promptIds, sampledTokens: sampled, turnId };
}

/** sampling helper: takes Float32Array-like probs, returns sampled index */
function sampleFromProbs(probs){
  // ensure non-negative and sum to 1 (they should)
  let sum = 0;
  for(let i=0;i<probs.length;i++){ if(!isFinite(probs[i]) || probs[i] < 0) probs[i]=0; sum += probs[i]; }
  if(sum === 0) return Math.floor(Math.random()*probs.length);
  let r = Math.random() * sum;
  for(let i=0;i<probs.length;i++){
    r -= probs[i];
    if(r <= 0) return i;
  }
  return probs.length - 1;
}

/**
 * applyRewardUpdate:
 * - Recomputes logits for each generation timestep under gradient tape
 * - Gathers the log probabilities of the previously sampled tokens (so log_probs are differentiable w.r.t model params)
 * - Loss = - reward * sum(log_probs)
 * - optimizer.minimize applied to perform one update step
 *
 * promptTokens: array length SEQ_LEN (numbers)
 * sampledTokens: array length MAX_GEN (numbers)
 */
export async function applyRewardUpdate(model, promptTokens, sampledTokens, rewardScalar=1.0){
  // small guard
  if(!model) throw new Error('model missing');

  // One optimizer step inside tf.tidy
  const vocabSize = model.outputs[0].shape[1];

  // Use optimizer.minimize to perform gradients update (async-friendly pattern)
  await optimizer.minimize(() => {
    // We need to accumulate log_probs as differentiable tensors
    let ctx = promptTokens.slice(); // copy
    let logProbsAcc = null;

    // For each sampled token, compute logits and log-softmax, then gather
    for(let t=0;t<sampledTokens.length;t++){
      const inputArr = tf.tensor2d([ctx], [1, SEQ_LEN], 'int32');
      const logits = model.apply(inputArr); // [1, vocabSize]
      // compute log softmax
      const logsoft = tf.logSoftmax(logits);
      // gather logprob of the sampled token index
      const idx = sampledTokens[t];
      // gather at [0, idx]
      const logp = tf.squeeze(tf.gather(logsoft, tf.tensor1d([idx], 'int32'), 1)); // tricky: gather along axis? simpler: slice
      // Alternative safe approach: use logits.flatten() and pick offset idx
      // But above works in many tf.js builds. To be safe, do slice:
      // const logp = tf.squeeze(logsoft.slice([0, idx], [1,1]));
      // accumulate
      logProbsAcc = logProbsAcc ? tf.add(logProbsAcc, logp) : logp;
      // advance ctx by pushing sampled token (drop first)
      ctx = ctx.slice(1).concat([idx]);
      tf.dispose([inputArr, logits, logsoft]); // keep only logp
    }

    // final loss = - reward * sum(log_probs)
    // If no logProbsAcc (shouldn't happen) make zero
    const totalLogProb = logProbsAcc ? logProbsAcc : tf.scalar(0);
    const loss = tf.mul(tf.scalar(-rewardScalar), totalLogProb);
    // free
    return loss;
  }, /* returnCost */ true);

  // allow garbage collection and UI responsiveness
  await tf.nextFrame();
}
