// แผงเทรน KB บนหน้า /logs/view — ส่งงานให้เซิร์ฟเวอร์รัน build_kb.py แล้วโพลล์สถานะ
(function () {
  const $ = (id) => document.getElementById(id);
  let timer = null;

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(refresh, 4000);
  }

  function tailLines(lines) {
    return (lines || []).slice(-14).join('\n');
  }

  function render(s) {
    const box = $('trainStatus');
    if (s.running) {
      box.className = 'train-status';
      box.textContent =
        `⏳ กำลังเทรน KB "${s.code}" จาก ${s.url} (เริ่ม ${s.started_at})\n` +
        tailLines(s.lines) +
        `\n(ปิดหน้านี้ได้ เทรนทำงานต่อในเซิร์ฟเวอร์ — กลับมาเปิดหน้านี้เพื่อดูต่อ)`;
      schedule();
    } else if (s.started_at && s.returncode === 0) {
      box.className = 'train-status ok';
      box.textContent =
        `✅ เทรน KB "${s.code}" เสร็จสิ้น (${s.finished_at})\n` +
        tailLines(s.lines) +
        `\n→ รีเฟรชหน้าแชท http://127.0.0.1:8080 จะเห็น KB ใหม่ในแถบซ้าย`;
      $('trainBtn').disabled = false;
    } else if (s.started_at && s.returncode !== null && s.returncode !== 0) {
      box.className = 'train-status err';
      box.textContent =
        `❌ เทรนล้มเหลว (exit ${s.returncode}) ${s.error || ''}\n` + tailLines(s.lines);
      $('trainBtn').disabled = false;
    } else {
      // ยังไม่มีงาน — คงข้อความเริ่มต้นไว้
      $('trainBtn').disabled = false;
    }
  }

  async function refresh() {
    try {
      const res = await fetch('/kb/build/status');
      render(await res.json());
    } catch (e) {
      schedule();
    }
  }

  document.getElementById('trainBtn').addEventListener('click', async () => {
    const btn = $('trainBtn');
    btn.disabled = true;
    try {
      const res = await fetch('/kb/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: $('trainUrl').value.trim(),
          company_code: $('trainCode').value.trim(),
          max_pages: parseInt($('trainPages').value || '15', 10),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      $('trainStatus').className = 'train-status';
      $('trainStatus').textContent =
        `🚀 เริ่มเทรน KB "${data.company_code}" แล้ว (${data.max_pages} หน้าสูงสุด)\nความคืบหน้าจะไหลมาที่นี่...`;
      refresh();
    } catch (err) {
      $('trainStatus').className = 'train-status err';
      $('trainStatus').textContent = '❌ ' + err.message;
      btn.disabled = false;
    }
  });

  refresh();
})();
