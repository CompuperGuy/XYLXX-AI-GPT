// ui.js - light helpers (escape, small dom)
export function createBubble(text, who){
  const c = document.createElement('div'); c.className = 'msg ' + who; c.textContent = text; return c;
}
export function escapeHtml(s){ return s.replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
