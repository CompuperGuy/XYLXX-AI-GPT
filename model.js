// model.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

export async function loadVocab(url) {
  const res = await fetch(url);
  return await res.json();
}

export function buildModel(vocabSize, embedDim=64, numHeads=2) {
  const input = tf.input({shape: [10]});
  const embed = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim }).apply(input);

  const query = tf.layers.dense({units: embedDim})(embed);
  const key = tf.layers.dense({units: embedDim})(embed);
  const value = tf.layers.dense({units: embedDim})(embed);
  const scores = tf.layers.dot({axes: -1})([query, key]);
  const weights = tf.layers.activation({activation: "softmax"}).apply(scores);
  const context = tf.layers.dot({axes: [2,1]})([weights, value]);

  const flat = tf.layers.flatten().apply(context);
  const dense = tf.layers.dense({ units: 128, activation: "relu" }).apply(flat);
  const output = tf.layers.dense({ units: vocabSize, activation: "softmax" }).apply(dense);

  const model = tf.model({inputs: input, outputs: output});
  model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy" });
  return model;
}

export async function generateReply(model, vocab, inputText) {
  const tokens = inputText.toLowerCase().split(/\s+/).map(w => vocab[w] ?? 0);
  while (tokens.length < 10) tokens.push(0);
  const xs = tf.tensor2d([tokens.slice(0,10)], [1,10]);
  const preds = model.predict(xs);
  const idx = preds.argMax(-1).dataSync()[0];
  const word = Object.keys(vocab).find(k => vocab[k] === idx) || "…";
  return "XYLXX: " + word;
}

export function rewardTrain(model, vocab, text, positive=true) {
  const tokens = text.toLowerCase().split(/\s+/).map(w => vocab[w] ?? 0);
  while (tokens.length < 10) tokens.push(0);
  const xs = tf.tensor2d([tokens.slice(0,10)], [1,10]);
  const ys = tf.tensor2d([tokens[0]], [1,1]);
  model.fit(xs, ys, {epochs: 1});
}
