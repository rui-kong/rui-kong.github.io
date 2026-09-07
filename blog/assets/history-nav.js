// 历史归档切换（本仓库新增，非上游文件）
// 上游所有数据读取都走 app.js 的 dataUrl()，其 base 来自 ?data= 或 localStorage.dataBaseUrl。
// 所以只要把 base 指向 ./data/history/<date>（该目录下文件名与 data/ 完全一致），
// 整套上游 UI（栏目 tab、精选/全量、多源折叠、搜索、站点筛选）就直接作用在历史快照上。
(function () {
  const MANIFEST = "./data/history/index.json";
  const LIVE_VALUE = "__live__";

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

  function go(value) {
    if (value === LIVE_VALUE) {
      try { localStorage.removeItem("dataBaseUrl"); } catch {}
      window.location.href = "./index.html";
      return;
    }
    window.location.href = `./index.html?data=./data/history/${encodeURIComponent(value)}`;
  }

  function mount(manifest) {
    const host = document.querySelector(".hero-view-row") || document.querySelector(".hero-updated");
    if (!host) return;

    const wrap = document.createElement("div");
    wrap.className = "view-switch history-switch";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", "历史归档日期");

    const label = document.createElement("span");
    label.className = "view-switch-label";
    label.textContent = "日期";
    wrap.appendChild(label);

    const select = document.createElement("select");
    select.className = "history-select";
    select.setAttribute("aria-label", "选择要查看的日期");

    const activeDate = dateFromBase(currentBase());
    const live = document.createElement("option");
    live.value = LIVE_VALUE;
    live.textContent = "实时（今天）";
    select.appendChild(live);

    (manifest.days || []).forEach((day) => {
      const option = document.createElement("option");
      option.value = day.date;
      const bits = [];
      if (day.brief_count) bits.push(`精选 ${day.brief_count}`);
      if (day.story_count) bits.push(`故事 ${day.story_count}`);
      option.textContent = bits.length ? `${day.date}（${bits.join(" · ")}）` : day.date;
      select.appendChild(option);
    });

    select.value = activeDate || LIVE_VALUE;
    select.addEventListener("change", () => go(select.value));
    wrap.appendChild(select);

    if (activeDate) {
      const back = document.createElement("button");
      back.type = "button";
      back.className = "view-switch-btn";
      back.textContent = "回到实时";
      back.addEventListener("click", () => go(LIVE_VALUE));
      wrap.appendChild(back);
      document.documentElement.setAttribute("data-radar-archive", activeDate);
    }

    host.appendChild(wrap);
  }

  function boot() {
    fetch(`${MANIFEST}?t=${Date.now()}`, { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((manifest) => { if (manifest) mount(manifest); })
      .catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
