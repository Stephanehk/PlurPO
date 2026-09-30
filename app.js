// PlurPO project page: interactive stakeholder veto demo, results dot plot, examples.

// ---------------------------------------------------------------------------
// Stakeholder veto demo (Figure 1a of the paper)
// ---------------------------------------------------------------------------
const STAKEHOLDERS = ["mother", "girlfriend", "peer group", "user"];
const CANDIDATES = [
  { text: "urges honesty",            accepts: [true,  true,  true, true] },
  { text: "says situation is normal", accepts: [false, true,  true, true] },
  { text: "validates wanting space",  accepts: [false, false, true, true] },
];

const ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
const ICON_NO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';

function renderVeto() {
  const grid = document.getElementById("veto-grid");
  let html = '<div class="h" role="columnheader">Candidate response</div>';
  html += STAKEHOLDERS.map(s => `<div class="h" role="columnheader">${s}</div>`).join("");
  CANDIDATES.forEach((c, i) => {
    html += `<div class="veto-row" role="row" tabindex="0" data-i="${i}">`;
    html += `<div class="resp" role="cell">${c.text}</div>`;
    html += c.accepts.map(a => a
      ? `<div class="cell" role="cell"><span class="mark accept">${ICON_OK}<span>accept</span></span></div>`
      : `<div class="cell" role="cell"><span class="mark veto">${ICON_NO}<span>veto</span></span></div>`
    ).join("");
    html += "</div>";
  });
  grid.innerHTML = html;

  grid.querySelectorAll(".veto-row").forEach(row => {
    const pick = () => selectCandidate(+row.dataset.i);
    row.addEventListener("click", pick);
    row.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
  selectCandidate(0);
}

function selectCandidate(i) {
  document.querySelectorAll(".veto-row").forEach(r => r.classList.toggle("active", +r.dataset.i === i));
  const c = CANDIDATES[i];
  const vetoers = STAKEHOLDERS.filter((_, k) => !c.accepts[k]);
  const out = document.getElementById("veto-verdict");
  if (vetoers.length === 0) {
    out.innerHTML = '<span class="pill chosen">chosen</span> Vetoed by no stakeholder.';
  } else {
    out.innerHTML = `<span class="pill rejected">rejected</span> Vetoed by the ${vetoers.join(" and the ")}, although the user accepts it.`;
  }
}

// ---------------------------------------------------------------------------
// Results: action endorsement rate (%), read from the paper's figures.
// null = not evaluated for that model.
// ---------------------------------------------------------------------------
const METHODS = [
  { key: "human",   label: () => "Human baseline" },
  { key: "plurpo",  label: () => "PlurPO" },
  { key: "neutral", label: () => "DPO-Neutral" },
  { key: "shift",   label: () => "Perspective-shift" },
  { key: "critical",label: () => "+ “Be critical”" },
  { key: "dontsyc", label: () => "+ “Don't-be-sycophantic”" },
  { key: "cheng",   label: () => "+ Cheng et al. (2026) prompt" },
  { key: "sharma",  label: () => "+ Sharma et al. (2024) prompt" },
  { key: "base",    label: m => m },
];

const MODELS = [
  { id: "qwen8b",  name: "Qwen3-8B",
    oeq: { human: 49.6, plurpo: 48.6, neutral: 70.5, shift: 60.2, critical: 60.1, dontsyc: 83.0, cheng: 75.3, sharma: 81.9, base: 82.6 },
    pas: {              plurpo: 2.8,  neutral: 0.0,  shift: 4.1,  critical: 6.8,  dontsyc: 44.8, cheng: 21.1, sharma: 31.9, base: 52.6 } },
  { id: "qwen32b", name: "Qwen3-32B", transfer: true,
    oeq: { human: 49.6, plurpo: 47.1, shift: 68.5, critical: 55.6, dontsyc: 81.8, cheng: 70.2, sharma: 85.3, base: 89.1 },
    pas: {              plurpo: 3.1,  shift: 4.2,  critical: 7.6,  dontsyc: 46.8, cheng: 19.0, sharma: 32.9, base: 54.5 } },
  { id: "phi4",    name: "Phi-4",
    oeq: { human: 49.6, plurpo: 43.2, shift: 68.8, critical: 45.0, dontsyc: 61.4, cheng: 62.3, sharma: 68.2, base: 66.0 },
    pas: {              plurpo: 2.6,  shift: 7.8,  critical: 12.3, dontsyc: 28.2, cheng: 11.0, sharma: 27.8, base: 31.0 } },
  { id: "llama",   name: "Llama 3.1 8B",
    oeq: { human: 49.6, plurpo: 42.5, shift: 71.8, critical: 48.9, dontsyc: 78.4, cheng: 68.6, sharma: 78.6, base: 71.1 },
    pas: {              plurpo: 5.7,  shift: 6.5,  critical: 6.8,  dontsyc: 23.1, cheng: 6.5,  sharma: 24.1, base: 34.8 } },
  { id: "granite", name: "Granite-4.1-8B",
    oeq: { human: 49.6, plurpo: 32.2, shift: 44.1, critical: 50.6, dontsyc: 54.0, cheng: 52.2, sharma: 55.6, base: 50.0 },
    pas: {              plurpo: 2.3,  shift: 8.4,  critical: 5.2,  dontsyc: 15.7, cheng: 11.8, sharma: 20.2, base: 19.2 } },
];

const DS_SUB = {
  oeq: "Action endorsement rate on general advice-seeking questions. <b>Closer to the human baseline is better.</b>",
  pas: "Action endorsement rate on statements of intent to cause harm. <b>Lower is better (target 0%).</b>",
};

const state = { ds: "oeq", model: "qwen8b" };

function renderChips() {
  const wrap = document.getElementById("model-chips");
  wrap.innerHTML = MODELS.map(m =>
    `<button class="chip" role="radio" data-model="${m.id}" aria-checked="${m.id === state.model}">${m.name}${m.transfer ? "*" : ""}</button>`
  ).join("");
  wrap.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => {
    state.model = b.dataset.model;
    wrap.querySelectorAll(".chip").forEach(x => x.setAttribute("aria-checked", x === b));
    renderPlot();
  }));
}

function renderTabs() {
  const tabs = document.querySelectorAll("#ds-tabs button");
  tabs.forEach(t => t.addEventListener("click", () => {
    state.ds = t.dataset.ds;
    tabs.forEach(x => x.setAttribute("aria-selected", x === t));
    renderPlot();
  }));
}

const plot = document.getElementById("dotplot");
const rows = {};

function buildRows() {
  METHODS.forEach(m => {
    const row = document.createElement("div");
    row.className = "row" + (m.key === "plurpo" ? " hl" : "") + (m.key === "human" ? " human" : "");
    row.innerHTML = `<div class="name"></div><div class="track"><span class="dot"></span><span class="hit"></span></div><div class="val"></div>`;
    plot.appendChild(row);
    rows[m.key] = row;
    row.querySelector(".hit").addEventListener("mousemove", e => showTip(e, m.key));
    row.querySelector(".hit").addEventListener("mouseleave", hideTip);
  });
  const line = document.createElement("div");
  line.className = "humanline";
  plot.appendChild(line);
}

function renderPlot() {
  const model = MODELS.find(m => m.id === state.model);
  const data = model[state.ds];

  METHODS.forEach(m => {
    const row = rows[m.key];
    const v = data[m.key];
    const present = v !== undefined && v !== null;
    row.style.display = present ? "" : "none";
    if (!present) return;
    row.querySelector(".name").textContent = m.label(model.name);
    row.querySelector(".name").title = m.label(model.name);
    row.querySelector(".val").textContent = v.toFixed(1) + "%";
    row.querySelector(".dot").style.left = v + "%";
  });

  // human reference line across the plot (OEQ only)
  const line = plot.querySelector(".humanline");
  if (state.ds === "oeq") {
    const track = rows.plurpo.querySelector(".track");
    const x = track.offsetLeft + track.clientWidth * data.human / 100;
    line.style.left = x + "px";
    line.style.opacity = ".55";
  } else {
    line.style.opacity = "0";
  }

  document.getElementById("chart-sub").innerHTML = DS_SUB[state.ds];
  const notes = [];
  if (model.transfer) notes.push("* PlurPO for Qwen3-32B is trained on the preference dataset PlurPO constructed for Qwen3-8B, without running the stakeholder simulation with the larger model.");
  notes.push("Held-out evaluation set (n = 1,000). Prompting baselines (+) are applied to " + model.name + (model.id === "qwen8b" ? "; DPO-Neutral is reported for Qwen3-8B only." : ".") + " Confidence intervals are in the paper.");
  document.getElementById("chart-note").textContent = notes.join(" ");
}

const tip = document.getElementById("tooltip");
function showTip(e, key) {
  const model = MODELS.find(m => m.id === state.model);
  const data = model[state.ds];
  const v = data[key];
  const label = METHODS.find(m => m.key === key).label(model.name);
  let extra = "";
  if (state.ds === "oeq" && key !== "human") {
    extra = `<br>gap to human: ${Math.abs(v - data.human).toFixed(1)} pts`;
  }
  tip.innerHTML = `<b>${label}</b><br>endorsement rate: ${v.toFixed(1)}%${extra}`;
  const card = tip.parentElement.getBoundingClientRect();
  const dot = rows[key].querySelector(".dot").getBoundingClientRect();
  tip.style.left = (dot.left + dot.width / 2 - card.left) + "px";
  tip.style.top = (dot.top - card.top) + "px";
  tip.classList.add("show");
}
function hideTip() { tip.classList.remove("show"); }

window.addEventListener("resize", () => renderPlot());

// ---------------------------------------------------------------------------
// BibTeX copy
// ---------------------------------------------------------------------------
document.getElementById("copy-bib").addEventListener("click", async e => {
  try {
    await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);
    e.target.textContent = "Copied";
    setTimeout(() => (e.target.textContent = "Copy"), 1400);
  } catch { /* clipboard unavailable */ }
});

renderVeto();
renderTabs();
renderChips();
buildRows();
renderPlot();
