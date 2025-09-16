// firebase.js (modular client SDK)
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";
import { getStorage, ref, uploadBytes } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-storage.js";

const firebaseConfig = {
  apiKey: "AIzaSyALx3RDkGf27Nv-mcatszH8Dx5wuPA1jF0",
  authDomain: "ai-bot-f9965.firebaseapp.com",
  projectId: "ai-bot-f9965",
  storageBucket: "ai-bot-f9965.firebasestorage.app",
  messagingSenderId: "669889318286",
  appId: "1:669889318286:web:ccd12b68d7c0c51b29a8df",
  measurementId: "G-120MFDQRNJ"
};

let db=null, storage=null;
export async function initFirebase(){
  const app = initializeApp(firebaseConfig);
  db = getFirestore(app);
  storage = getStorage(app);
}

/* save chat entry */
export async function saveChatEvent(obj){
  if(!db) return;
  try{ await addDoc(collection(db, 'chats'), { user: obj.user, bot: obj.bot, turnId: obj.turnId, ts: obj.ts || Date.now() }); }
  catch(e){ console.warn('saveChatEvent', e); }
}

/* save reward event */
export async function saveRewardEvent(turnId, reward){
  if(!db) return;
  try{ await addDoc(collection(db, 'rewards'), { turnId, reward, ts: Date.now() }); }
  catch(e){ console.warn('saveRewardEvent', e); }
}

/* helper used by model.js to upload weights */
export async function uploadModelArtifactsToFirebase(path, blob){
  if(!storage) throw new Error('Storage not initialized');
  try{
    const r = ref(storage, path);
    await uploadBytes(r, blob);
    return true;
  }catch(e){ console.warn('uploadModelArtifactsToFirebase', e); throw e; }
}
