/* echoSentinel console — shared shell, API helpers, formatting. */

const CLASSES = {
  vessel:              { label: "VESSEL",        color: "#00f0ff" },
  marine_animal:       { label: "MARINE ANIMAL", color: "#34f5c5" },
  natural_sound:       { label: "NATURAL SOUND", color: "#38bdf8" },
  other_anthropogenic: { label: "ANTHROPOGENIC", color: "#ffb454" },
};
const CLASS_BY_ID = {
  1: "vessel", 2: "marine_animal", 3: "natural_sound", 4: "other_anthropogenic",
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

/* ---------- formatting ---------- */
function fmtTime(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function fmtDate(epoch) {
  return new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
}
function fmtKHz(sr) { return sr ? (sr / 1000).toFixed(sr % 1000 ? 1 : 0) + " kHz" : "—"; }

function chip(className, extra = "") {
  const c = CLASSES[className];
  if (!c) return "";
  return `<span class="chip" data-class="${className}"><i></i>${c.label}${extra}</span>`;
}

function countUp(el, target, decimals = 0, duration = 900) {
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = from + (target - from) * eased;
    el.textContent = decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString();
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ---------- shared shell ---------- */
const NAV = [
  { href: "index.html",   key: "overview", label: "OVRVW",
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/></svg>' },
  { href: "analyze.html", key: "analyze", label: "ANLYZ",
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 12h3l2-6 4 12 3-9 2 3h6"/></svg>' },
  { href: "archive.html", key: "archive", label: "ARCHV",
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="4" width="18" height="5" rx="1"/><path d="M5 9v10a1 1 0 001 1h12a1 1 0 001-1V9M10 13h4"/></svg>' },
];

function renderShell(activeKey) {
  const banner = `<div class="class-banner">Unclassified &nbsp;//&nbsp; For Training and Evaluation</div>`;
  const rail = `
    <nav class="rail">
      <div class="logo" title="echoSentinel">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="2.5"/><path d="M7.5 7.5a6.4 6.4 0 000 9M16.5 7.5a6.4 6.4 0 010 9"/>
          <path d="M4.7 4.7a10.3 10.3 0 000 14.6M19.3 4.7a10.3 10.3 0 010 14.6"/>
        </svg>
      </div>
      ${NAV.map(n => `<a href="${n.href}" class="${n.key === activeKey ? "active" : ""}">${n.icon}<span>${n.label}</span></a>`).join("")}
    </nav>`;
  const topbar = `
    <header class="topbar">
      <div class="brand">ECHOSENTINEL<small>UNDERWATER DOMAIN AWARENESS</small></div>
      <div class="status">
        <span id="model-led"><span class="led err"></span>MODEL</span>
        <span id="sys-version"></span>
        <span id="utc-clock">--:--:--Z</span>
      </div>
    </header>`;
  document.body.insertAdjacentHTML("afterbegin", banner + rail + topbar);

  const clock = document.getElementById("utc-clock");
  setInterval(() => {
    clock.textContent = new Date().toISOString().slice(11, 19) + "Z";
  }, 1000);
  clock.textContent = new Date().toISOString().slice(11, 19) + "Z";

  api("/api/system")
    .then(sys => {
      document.getElementById("model-led").innerHTML =
        `<span class="led ok"></span>${sys.model.model_name.toUpperCase()} ONLINE`;
      document.getElementById("sys-version").textContent = "V" + sys.version;
      document.dispatchEvent(new CustomEvent("system", { detail: sys }));
    })
    .catch(() => {
      document.getElementById("model-led").innerHTML =
        `<span class="led err"></span>MODEL OFFLINE`;
    });
}

/* staggered entrance indices */
function stagger(selector = ".rise") {
  document.querySelectorAll(selector).forEach((el, i) => el.style.setProperty("--i", i));
}
