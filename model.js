// model.js
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

export async function loadVocab(url) {
  const res = await fetch(url);
  return await res.json();
}

export function buildModel(vocabSize, embedDim=64) {
  const input = tf.input({shape: [10]});
  const embed = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim }).apply(input);
  const flat = tf.layers.flatten().apply(embed);
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
  const word = Object.keys(vocab).find(k => vocab[k] === idx) || null;

  // If neural net fails, fallback to web search
  if (!word || word === "…") {
    const crawl = await webSearch(inputText);
    return crawl ? `XYLXX (web): ${crawl}` : "XYLXX: I don’t know.";
  }

  return "XYLXX: " + word;
}

// Auto web search via DuckDuckGo proxy (Google blocks CORS)
async function webSearch(query) {
  try {
    const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const res = await fetch("https://api.allorigins.win/get?url=" + encodeURIComponent(url));
    const data = await res.json();
    const html = data.contents;

    // Extract snippet text
    const match = html.match(/<a[^>]+class="result__snippet"[^>]*>(.*?)<\/a>/i);
    if (match) {
      return match[1].replace(/<[^>]+>/g, "");
    }
  } catch (e) {
    console.error("Web search failed:", e);
  }
  return null;
}

export function rewardTrain(model, vocab, text, positive=true) {
  const tokens = text.toLowerCase().split(/\s+/).map(w => vocab[w] ?? 0);
  while (tokens.length < 10) tokens.push(0);
  const xs = tf.tensor2d([tokens.slice(0,10)], [1,10]);
  const ys = tf.tensor2d([tokens[0]], [1,1]);
  model.fit(xs, ys, {epochs: 1});
}
