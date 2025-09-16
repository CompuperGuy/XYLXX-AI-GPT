import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getFirestore, collection, addDoc } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyALx3RDkGf27Nv-mcatszH8Dx5wuPA1jF0",
  authDomain: "ai-bot-f9965.firebaseapp.com",
  projectId: "ai-bot-f9965",
  storageBucket: "ai-bot-f9965.firebasestorage.app",
  messagingSenderId: "669889318286",
  appId: "1:669889318286:web:ccd12b68d7c0c51b29a8df",
  measurementId: "G-120MFDQRNJ"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export async function saveChat(user, bot, id) {
  await addDoc(collection(db, "chats"), { user, bot, id, ts: Date.now() });
}

export async function saveReward(id, score) {
  await addDoc(collection(db, "rewards"), { id, score, ts: Date.now() });
}
