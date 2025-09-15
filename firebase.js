// firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

/* Your provided firebaseConfig */
const firebaseConfig = {
  apiKey: "AIzaSyALx3RDkGf27Nv-mcatszH8Dx5wuPA1jF0",
  authDomain: "ai-bot-f9965.firebaseapp.com",
  projectId: "ai-bot-f9965",
  storageBucket: "ai-bot-f9965.firebasestorage.app",
  messagingSenderId: "669889318286",
  appId: "1:669889318286:web:ccd12b68d7c0c51b29a8df",
  measurementId: "G-120MFDQRNJ"
};

let db = null;

export async function initFirebase(){
  try{
    const app = initializeApp(firebaseConfig);
    db = getFirestore(app);
    console.log('Firebase init done');
  }catch(e){
    console.warn('Firebase init error', e);
  }
}

export async function saveChatEvent(obj){
  if(!db) return;
  try{
    await addDoc(collection(db, 'chats'), {
      user: obj.user, bot: obj.bot, turnId: obj.turnId, ts: obj.ts || Date.now()
    });
  }catch(e){ console.warn('saveChatEvent error', e); }
}

export async function saveRewardEvent(obj){
  if(!db) return;
  try{
    await addDoc(collection(db, 'rewards'), {
      turnId: obj.turnId, reward: obj.reward, ts: obj.ts || Date.now()
    });
  }catch(e){ console.warn('saveRewardEvent error', e); }
}
