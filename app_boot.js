// app_boot.js
import { initFirebase } from './firebase.js';
import { loadVocab, loadModelFromIndexedDB, makeModel, generateResponseAndRecord } from './model.js';
import { saveChatEvent, saveRewardEvent } from './firebase.js';

let state = { model: null, vocab: null, invVocab: null, initialized: false };

export async function initApp(opts){
  const { viewportEl, promptEl, sendBtn } = opts;
  // init firebase (client)
  await initFirebase();

  // load vocab and model (try IndexedDB first)
  const vocab = await loadVocab('./vocab.json');
  state.vocab = vocab;
  state.invVocab = Object.fromEntries(Object.entries(vocab).map(([k,v])=>[v,k]));

  let model = await loadModelFromIndexedDB();
  if(!model){
    const created = await makeModel(Object.keys(vocab).length);
    model = created;
    // save initial small model (non-blocking)
    model.save('indexeddb://xylxx-init').catch(()=>console.warn('initial save failed'));
  }
  state.model = model;
  state.initialized = true;

  // UI helpers
  function appendBubble(text, who, metaHtml=''){
    const container = document.createElement('div');
    container.className = 'msgRow ' + (who==='user' ? 'userRow' : 'botRow');
    const bubble = document.createElement('div');
    bubble.className = 'msg ' + who;
    bubble.innerHTML = text;
    container.appendChild(bubble);
    if(metaHtml){
      const meta = document.createElement('div');
      meta.className = 'metaRow';
      meta.innerHTML = metaHtml;
      container.appendChild(meta);
    }
    viewportEl.appendChild(container);
    viewportEl.scrollTop = viewportEl.scrollHeight;
  }

  // send handler
  async function onSend(){
    if(!state.initialized) return;
    const text = promptEl.value.trim();
    if(!text) return;
    promptEl.value = '';
    appendBubble(escapeHtml(text), 'user');

    // generate response (also records sample tokens and turnId)
    const { reply, turnId, promptTokens, sampledTokens } = await generateResponseAndRecord(state.model, text, state.vocab, state.invVocab);
    // build meta html with votes wired to functions
    const metaHtml = `<div class="small">XYLXX GPT-1</div>
      <div class="small" style="margin-left:12px">
        <button class="vote" id="up_${turnId}">👍</button>
        <button class="vote" id="down_${turnId}">👎</button>
      </div>`;
    appendBubble(escapeHtml(reply), 'bot', metaHtml);

    // attach events after element is in DOM
    setTimeout(()=>{
      const up = document.getElementById('up_'+turnId);
      const down = document.getElementById('down_'+turnId);
      if(up) up.onclick = async ()=>{
        await saveRewardEvent(turnId, 1);
        await state.model.applyRewardUpdate(promptTokens, sampledTokens, +1);
        appendBubble('Thanks — model updated (positive).', 'sys');
      };
      if(down) down.onclick = async ()=>{
        await saveRewardEvent(turnId, -1);
        await state.model.applyRewardUpdate(promptTokens, sampledTokens, -1);
        appendBubble('Thanks — model updated (negative).', 'sys');
      };
    }, 80);

    // save chat record
    saveChatEvent({ user: text, bot: reply, turnId, ts: Date.now() }).catch(()=>{});
  }

  // wire send button and enter
  sendBtn.addEventListener('click', onSend);
  promptEl.addEventListener('keydown', (e) => {
    if(e.key==='Enter' && !e.shiftKey){
      e.preventDefault();
      onSend();
    }
  });

  // expose simple debug helpers
  window.xylxx_state = state;
  window.xylxx_save_model = async (name='xylxx-model')=>{
    await state.model.save(`indexeddb://${name}`);
    return true;
  };
}

function escapeHtml(s){ return s.replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
