import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js";

// ----------------------------
// Vocabulary loader
// ----------------------------
export async function loadVocab(url) {
  const res = await fetch(url);
  return res.json();
}

// ----------------------------
// Build an advanced model
// ----------------------------
export function buildModel(vocabSize) {
  const input = tf.input({ shape: [50] });

  // Embedding Layer
  let x = tf.layers.embedding({ inputDim: vocabSize, outputDim: 128 })(input);

  // BiLSTM
  x = tf.layers.bidirectional({
    layer: tf.layers.lstm({ units: 128, returnSequences: true }),
    mergeMode: "concat"
  }).apply(x);

  // Attention Mechanism
  const attention = tf.layers.dense({ units: 1, activation: "tanh" }).apply(x);
  const attentionWeights = tf.layers.softmax({ axis: 1 }).apply(attention);
  const context = tf.layers.dot({ axes: [1, 1] }).apply([attentionWeights, x]);

  // Dense Layers
  x = tf.layers.flatten().apply(context);
  x = tf.layers.dense({ units: 256, activation: "relu" }).apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);

  const output = tf.layers.dense({ units: vocabSize, activation: "softmax" }).apply(x);

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

// ----------------------------
// Reply Generator
// ----------------------------
export async function generateReply(model, vocab, inputText) {
  const words = inputText.toLowerCase().split(/\s+/);
  const indices = words.map(w => vocab[w] || 0);
  const padded = indices.concat(Array(50 - indices.length).fill(0)).slice(0, 50);

  const input = tf.tensor2d([padded], [1, 50]);
  const prediction = model.predict(input);
  const idx = prediction.argMax(-1).dataSync()[0];

  const invVocab = Object.entries(vocab).reduce((acc, [w, i]) => {
    acc[i] = w;
    return acc;
  }, {});

  const word = invVocab[idx] || "hmm...";
  input.dispose();
  prediction.dispose();
  return word;
}

// ----------------------------
// Reward Training
// ----------------------------
export function rewardTrain(model, vocab, text, positive) {
  const indices = text.toLowerCase().split(/\s+/).map(w => vocab[w] || 0);
  const padded = indices.concat(Array(50 - indices.length).fill(0)).slice(0, 50);
  const input = tf.tensor2d([padded], [1, 50]);

  const label = tf.tensor2d([[positive ? 1 : 0]], [1, 1]);

  model.fit(input, label, { epochs: 1, verbose: 0 }).then(() => {
    input.dispose();
    label.dispose();
  });
}
