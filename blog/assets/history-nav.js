// 历史归档导航（本仓库新增，非上游文件）
// 上游 app.js 的所有数据读取都走 dataUrl()，base 取自 ?data= 或 localStorage.dataBaseUrl，
// 所以只要把 base 指向 ./data/history/<date>（目录内文件名与 data/ 一致），
// 整套上游 UI（栏目 tab、精选/全量、多源折叠、搜索、站点筛选）就作用在历史快照上。
(function () {
  const MANIFEST = "./data/history/index.json";

  const state = { days: [], index: -1, byDate: new Map() };

  function currentBase() {
    try {
      const fromQuery = new URLSearchParams(window.location.search).get("data");
      if (fromQuery) return fromQuery.trim().replace(/\/+$/, "");
      return (localStorage.getItem("dataBaseUrl") || "").trim().replace(/\/+$/, "");
    } catch {
      return "";
    }
  }

  function dateFromBase(base) {
    const match = /history\/(\d{4}-\d{2}-\d{2})$/.exec(base || "");
    return match ? match[1] : "";
  }

  function goLive() {
    try { localStorage.removeItem("dataBaseUrl"); } catch {}
    window.location.href = "./index.html";
  }

  function goDate(date) {
    if (!date) return;
    window.location.href = `./index.html?data=./data/history/${encodeURIComponent(date)}`;
  }

  // 日历里挑到没有归档的日期时，落到最近的一天，而不是报错。
  function nearestDate(wanted) {
    if (state.byDate.has(wanted)) return wanted;
    let best = null;
    let bestGap = Infinity;
    const target = Date.parse(`${wanted}T00:00:00Z`);
    state.days.forEach((day) => {
      const gap = Math.abs(Date.parse(`${day.date}T00:00:00Z`) - target);
      if (gap < bestGap) { bestGap = gap; best = day.date; }
    });
    return best;
  }

  function dayLabel(day) {
    if (!day) return "";
    const bits = [];
    if (day.brief_count) bits.push(`精选 ${day.brief_count}`);
    if (day.story_count) bits.push(`故事 ${day.story_count}`);
    if (!bits.length && day.item_count) bits.push(`条目 ${day.item_count}`);
    return bits.join(" · ");
  }

  function button(text, title, onClick, extraClass) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = `view-switch-btn${extraClass ? ` ${extraClass}` : ""}`;
    el.textContent = text;
    if (title) el.title = title;
    el.addEventListener("click", onClick);
    return el;
  }

  function step(offset) {
    // days 是按日期倒序的：offset=+1 更早，-1 更晚。
    if (state.index < 0) {
      // 实时视图下只能往更早走，落到最新的一天归档。
      if (offset > 0 && state.days.length) goDate(state.days[0].date);
      return;
    }
    const next = state.days[state.index + offset];
    if (next) goDate(next.date);
  }

  function mount() {
    const host = document.querySelector(".hero-view-row") || document.querySelector(".hero-updated");
    if (!host) return;

    const activeDate = dateFromBase(currentBase());
    state.index = activeDate ? state.days.findIndex((day) => day.date === activeDate) : -1;
    const active = state.index >= 0 ? state.days[state.index] : null;
    const newest = state.days[0];
    const oldest = state.days[state.days.length - 1];

    const wrap = document.createElement("div");
    wrap.className = "view-switch history-nav";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", "历史归档日期");

    const label = document.createElement("span");
    label.className = "view-switch-label";
    label.textContent = "日期";
    wrap.appendChild(label);

    const live = button("实时", "回到实时数据（快捷键 T）", goLive, activeDate ? "" : "is-on");
    wrap.appendChild(live);

    const prev = button("‹", "更早一天（← 键）", () => step(1), "history-step");
    prev.disabled = state.index >= 0 && state.index >= state.days.length - 1;
    wrap.appendChild(prev);

    // 原生日期控件：点一下出日历，可直接跳月，比长下拉快得多。
    const picker = document.createElement("input");
    picker.type = "date";
    picker.className = "history-date";
    picker.setAttribute("aria-label", "选择归档日期");
    if (oldest) picker.min = oldest.date;
    if (newest) picker.max = newest.date;
    picker.value = activeDate || (newest ? newest.date : "");
    picker.addEventListener("change", () => {
      const target = nearestDate(picker.value);
      if (target && target !== activeDate) goDate(target);
      else picker.value = activeDate || picker.value;
    });
    wrap.appendChild(picker);

    const next = button("›", "更晚一天（→ 键）", () => step(-1), "history-step");
    next.disabled = state.index <= 0;
    wrap.appendChild(next);

    const meta = document.createElement("span");
    meta.className = "history-meta";
    meta.textContent = active
      ? dayLabel(active)
      : `共 ${state.days.length} 天归档`;
    wrap.appendChild(meta);

    if (activeDate) document.documentElement.setAttribute("data-radar-archive", activeDate);
    host.appendChild(wrap);
  }

  function bindKeys() {
    document.addEventListener("keydown", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = (event.target && event.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "ArrowLeft") { event.preventDefault(); step(1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); step(-1); }
      else if (event.key === "t" || event.key === "T") { goLive(); }
    });
  }

  function boot() {
    fetch(`${MANIFEST}?t=${Date.now()}`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((manifest) => {
        if (!manifest || !Array.isArray(manifest.days) || !manifest.days.length) return;
        state.days = manifest.days;
        state.byDate = new Map(manifest.days.map((day) => [day.date, day]));
        mount();
        bindKeys();
      })
      .catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
