const els = {
  messages: document.getElementById('messages'),
  form: document.getElementById('chatForm'),
  input: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  stopBtn: document.getElementById('stopBtn'),
  statusText: document.getElementById('statusText'),
  providerMode: document.getElementById('providerMode'),
  providerRagMode: document.getElementById('providerRagMode'),
  providerUrl: document.getElementById('providerUrl'),
  providerModel: document.getElementById('providerModel'),
  reloadProvider: document.getElementById('reloadProvider'),
  clearChat: document.getElementById('clearChat'),
  copyLast: document.getElementById('copyLast'),
  enterToSend: document.getElementById('enterToSend'),
  saveHistory: document.getElementById('saveHistory'),
  kbList: document.getElementById('kbList'),
  mobileKbList: document.getElementById('mobileKbList'),
};

let controller = null;
let lastAssistantText = '';
let selectedKb = '';
const STORAGE_KEY = 'enterprise-rag-chat-history-v3';

const KB_PROMPTS = {
  demo: [
    'ลาเต้ราคาเท่าไหร่',
    'Café Siam มีกี่สาขา อยู่ที่ไหนบ้าง',
    'ค่าจัดส่งเท่าไหร่ สั่งขั้นต่ำเท่าไหร่',
    'ติดต่อร้านได้ยังไง',
  ],
  locktonwattana: [
    'D&O ประกันอะไรบ้าง',
    'ถ้าธุรกิจหยุดชะงัก มีประกันช่วยได้ไหม',
    'Lockton Wattana เป็นบริษัทอะไร',
  ],
  deves: [
    'เทเวศทำกิจกรรมอะไรบ้าง',
    'จะเดินทางไปเทเวศยังไง',
    'ดาวน์โหลดแบบฟอร์มเคลมได้ที่ไหน',
  ],
  mgcars: [
    'MG มีรถ EV รุ่นไหนบ้าง',
    'ราคา MG ZS เท่าไหร่',
    'ติดต่อ MG ได้ยังไง',
  ],
};

function updatePromptChips(code) {
  // greeting chips
  const greeting = document.getElementById('greeting');
  const greetingChips = document.getElementById('greetingChips');
  const greetingSub = document.getElementById('greetingSub');
  if (!greetingChips) return;

  const questions = KB_PROMPTS[code] || [];
  const label = Object.entries({
    demo: 'Café Siam (ร้านตัวอย่าง)',
    locktonwattana: 'Lockton Wattana',
    deves: 'Deves Insurance',
    mgcars: 'MG Cars Thailand',
  }).find(([k]) => k === code)?.[1] || code;

  if (greetingSub) greetingSub.textContent = label ? `ถามเกี่ยวกับ ${label} ได้เลย` : 'เลือก Knowledge Base แล้วถามได้เลย';

  greetingChips.innerHTML = questions.map(q =>
    `<button class="greeting-chip" type="button">${q}</button>`
  ).join('');
  greetingChips.querySelectorAll('.greeting-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      els.input.value = btn.textContent.trim();
      els.input.focus();
      els.form.requestSubmit();
    });
  });
}

function showGreeting() {
  const g = document.getElementById('greeting');
  if (g) g.classList.remove('hidden');
}

function hideGreeting() {
  const g = document.getElementById('greeting');
  if (g) g.classList.add('hidden');
}

function nowText() {
  return new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' });
}

function linkify(text) {
  const escaped = text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
  return escaped.replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
}

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) return '';
  const links = sources.map(s => {
    const pct = Math.round(s.similarity * 100);
    const title = s.title.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    return `<a class="source-link" href="${s.url}" target="_blank" rel="noopener noreferrer">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
      <span class="source-title">${title}</span>
      <span class="source-sim">${pct}%</span>
    </a>`;
  }).join('');
  return `<div class="sources"><span class="sources-label">แหล่งอ้างอิง</span><div class="sources-list">${links}</div></div>`;
}

function addMessage(role, text, save = true, sources = []) {
  hideGreeting();
  const item = document.createElement('div');
  item.className = `message ${role}`;
  const sourcesHtml = role === 'assistant' ? buildSourcesHtml(sources) : '';
  item.innerHTML = `
    <div class="meta">${role === 'user' ? 'คุณ' : 'AI'} • ${nowText()}</div>
    <div class="bubble">${linkify(text)}</div>
    ${sourcesHtml}
  `;
  // store sources so saveHistory() can persist them
  if (role === 'assistant' && sources.length) {
    item.dataset.sources = JSON.stringify(sources);
  }
  els.messages.appendChild(item);
  els.messages.scrollTop = els.messages.scrollHeight;

  if (role === 'assistant') lastAssistantText = text;
  if (save && els.saveHistory.checked) saveHistory();
  return item;
}

function addTyping() {
  const item = document.createElement('div');
  item.className = 'message assistant';
  item.innerHTML = `
    <div class="meta">AI • กำลังตอบ</div>
    <div class="bubble"><span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span></div>
  `;
  els.messages.appendChild(item);
  els.messages.scrollTop = els.messages.scrollHeight;
  return item;
}

function saveHistory() {
  const data = [...els.messages.querySelectorAll('.message')].map(m => ({
    role: m.classList.contains('user') ? 'user' : 'assistant',
    text: m.querySelector('.bubble')?.innerText || '',
    sources: m.dataset.sources ? JSON.parse(m.dataset.sources) : [],
  })).filter(x => x.text.trim() && !x.text.includes('กำลังตอบ'));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data.slice(-60)));
}

function loadHistory() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    // ไม่มีประวัติ — แสดง greeting (อยู่ใน HTML แล้ว)
    return;
  }
  try {
    const history = JSON.parse(raw);
    if (history.length === 0) return;
    history.forEach(m => addMessage(m.role, m.text, false, m.sources || []));
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

async function loadProvider() {
  try {
    els.statusText.textContent = 'กำลังโหลด provider...';
    const res = await fetch('/provider');
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    els.providerMode.textContent = data.provider || '-';
    els.providerRagMode.textContent = data.rag_mode || '-';
    els.providerUrl.textContent = data.base_url || '-';
    els.providerModel.textContent = data.chat_model || '-';
    els.statusText.textContent = `พร้อมใช้งานผ่าน ${data.provider || 'provider'}`;
  } catch (err) {
    els.statusText.textContent = 'โหลด provider ไม่สำเร็จ';
    els.providerMode.textContent = 'error';
    els.providerUrl.textContent = err.message;
  }
}

function selectKb(code, label) {
  selectedKb = code;
  // sync sidebar
  els.kbList.querySelectorAll('.kb-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.code === code)
  );
  // sync mobile pills
  els.mobileKbList.querySelectorAll('.mobile-kb-pill').forEach(b =>
    b.classList.toggle('active', b.dataset.code === code)
  );
  updateKbBadge(label);
  updatePromptChips(code);

  // clear chat และคืน greeting
  els.messages.innerHTML = '';
  localStorage.removeItem(STORAGE_KEY);
  els.messages.insertAdjacentHTML('afterbegin', `
    <div id="greeting" class="greeting">
      <div class="greeting-logo">RAG</div>
      <h2 class="greeting-title">สวัสดีครับ 👋</h2>
      <p class="greeting-sub" id="greetingSub">ถามได้เลย</p>
      <div class="greeting-chips" id="greetingChips"></div>
    </div>
  `);
  updatePromptChips(code);
}

async function loadKb() {
  try {
    const res = await fetch('/kb');
    const data = await res.json();
    selectedKb = data.default || '';
    els.kbList.innerHTML = '';
    els.mobileKbList.innerHTML = '';

    // ซ่อน KB ที่ยังไม่มีข้อมูล (docs=0 และ faqs=0) กันหลงกดแล้วบอทตอบไม่ได้
    const usableKbs = data.items.filter(kb => (kb.docs > 0 || kb.faqs > 0) || kb.code === selectedKb);

    usableKbs.forEach(kb => {
      const isActive = kb.code === selectedKb;

      // Sidebar button
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'kb-btn' + (isActive ? ' active' : '');
      btn.dataset.code = kb.code;
      btn.innerHTML = `
        <span>${kb.label}</span>
        <span class="kb-meta">${kb.docs.toLocaleString()} docs · ${kb.faqs.toLocaleString()} FAQs</span>
      `;
      btn.addEventListener('click', () => selectKb(kb.code, kb.label));
      els.kbList.appendChild(btn);

      // Mobile pill
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'mobile-kb-pill' + (isActive ? ' active' : '');
      pill.dataset.code = kb.code;
      pill.textContent = kb.label;
      pill.addEventListener('click', () => selectKb(kb.code, kb.label));
      els.mobileKbList.appendChild(pill);
    });

    const active = data.items.find(k => k.code === selectedKb);
    if (active) {
      updateKbBadge(active.label);
      updatePromptChips(active.code);
    }
  } catch {
    els.kbList.innerHTML = '<div class="kb-loading">โหลด KB ไม่สำเร็จ</div>';
    els.mobileKbList.innerHTML = '';
  }
}

function updateKbBadge(label) {
  let badge = document.getElementById('kbBadge');
  if (!badge) {
    badge = document.createElement('span');
    badge.id = 'kbBadge';
    badge.className = 'kb-badge';
    document.querySelector('.topbar h2').after(badge);
  }
  badge.textContent = label;
}

// ── Pipeline nodes: แสดงขั้นตอนการทำงานของระบบแบบกล่องต่อกัน ──
const PIPELINE_NODES = [
  { id: 'question', label: 'คำถาม',       sub: 'FastAPI' },
  { id: 'intent',   label: 'จับใจความ',   sub: 'Python' },
  { id: 'embed',    label: 'ฝังเวกเตอร์',  sub: 'Ollama bge-m3' },
  { id: 'search',   label: 'ค้นหาข้อมูล',  sub: 'Supabase pgvector' },
  { id: 'llm',      label: 'เขียนคำตอบ',   sub: 'Ollama qwen2.5:3b' },
  { id: 'sources',  label: 'อ้างอิง',      sub: 'แหล่งที่มา' },
];
const PIPELINE_ORDER = ['question', 'intent', 'embed', 'search', 'llm', 'sources'];

function renderPipeline() {
  const bar = document.getElementById('pipelineBar');
  if (!bar) return;
  bar.innerHTML = PIPELINE_NODES.map((n, i) => `
    <div class="pipe-node" id="node-${n.id}" data-step="${n.id}">
      <span class="pipe-label">${n.label}</span>
      <span class="pipe-sub">${n.sub}</span>
    </div>${i < PIPELINE_NODES.length - 1 ? '<span class="pipe-arrow">→</span>' : ''}
  `).join('');
}

function setPipelineState(activeStep) {
  // activeStep = node ที่กำลังทำงาน, ก่อนหน้า = done, หลัง = idle
  const idx = PIPELINE_ORDER.indexOf(activeStep);
  PIPELINE_ORDER.forEach((id, i) => {
    const el = document.getElementById(`node-${id}`);
    if (!el) return;
    el.classList.remove('done', 'active');
    if (idx === -1) return;
    if (i < idx) el.classList.add('done');
    else if (i === idx) el.classList.add('active');
  });
}

function resetPipeline() {
  PIPELINE_ORDER.forEach(id => {
    const el = document.getElementById(`node-${id}`);
    if (el) el.classList.remove('done', 'active');
  });
}

function finishPipeline() {
  PIPELINE_ORDER.forEach(id => {
    const el = document.getElementById(`node-${id}`);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
}

async function sendMessage(message) {
  controller = new AbortController();
  els.sendBtn.disabled = true;
  els.stopBtn.disabled = false;
  addMessage('user', message);
  const typing = addTyping();
  resetPipeline();
  setPipelineState('question');
  els.statusText.textContent = 'เริ่มประมวลผลคำถาม...';

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, company_code: selectedKb }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let streamedText = '';
    let bubbleStarted = false;
    let finalAnswer = '';
    let finalSources = [];

    const handleEvent = (evt) => {
      if (evt.type === 'status') {
        els.statusText.textContent = evt.label || '';
        setPipelineState(evt.step === 'nocontext' ? 'llm' : (evt.step === 'found' ? 'search' : evt.step));
      } else if (evt.type === 'token') {
        streamedText += evt.text || '';
        if (!bubbleStarted) {
          bubbleStarted = true;
          typing.querySelector('.meta').innerHTML = `AI • ${nowText()}`;
        }
        const bubble = typing.querySelector('.bubble');
        if (bubble) {
          bubble.innerHTML = linkify(streamedText) + '<span class="stream-cursor">▍</span>';
          els.messages.scrollTop = els.messages.scrollHeight;
        }
      } else if (evt.type === 'done') {
        finalAnswer = evt.answer || streamedText;
        finalSources = evt.sources || [];
      } else if (evt.type === 'error') {
        throw new Error(evt.detail || 'stream error');
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (line) handleEvent(JSON.parse(line));
      }
    }
    if (buf.trim()) handleEvent(JSON.parse(buf.trim()));

    typing.remove();
    addMessage('assistant', finalAnswer || streamedText || '(ไม่มีคำตอบ)', true, finalSources);
    finishPipeline();
    els.statusText.textContent = 'ตอบเสร็จแล้ว';
  } catch (err) {
    typing.remove();
    resetPipeline();
    addMessage('assistant', err.name === 'AbortError' ? 'หยุดการตอบแล้ว' : `เกิดข้อผิดพลาด: ${err.message}`);
    els.statusText.textContent = err.name === 'AbortError' ? 'หยุดแล้ว' : 'เกิดข้อผิดพลาด';
  } finally {
    els.sendBtn.disabled = false;
    els.stopBtn.disabled = true;
    controller = null;
    els.input.focus();
  }
}

els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  const message = els.input.value.trim();
  if (!message) return;
  els.input.value = '';
  sendMessage(message);
});

els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && els.enterToSend.checked) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});

els.input.addEventListener('input', () => {
  els.input.style.height = 'auto';
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + 'px';
});

els.stopBtn.addEventListener('click', () => controller?.abort());
els.reloadProvider.addEventListener('click', loadProvider);
els.clearChat.addEventListener('click', () => {
  els.messages.innerHTML = '';
  localStorage.removeItem(STORAGE_KEY);
  // คืน greeting กลับมา
  els.messages.insertAdjacentHTML('afterbegin', `
    <div id="greeting" class="greeting">
      <div class="greeting-logo">RAG</div>
      <h2 class="greeting-title">สวัสดีครับ 👋</h2>
      <p class="greeting-sub" id="greetingSub">ถามได้เลย</p>
      <div class="greeting-chips" id="greetingChips"></div>
    </div>
  `);
  updatePromptChips(selectedKb);
});
els.copyLast.addEventListener('click', async () => {
  if (!lastAssistantText) return;
  await navigator.clipboard.writeText(lastAssistantText);
  els.statusText.textContent = 'Copy คำตอบล่าสุดแล้ว';
});

renderPipeline();
loadHistory();
loadProvider();
loadKb();
