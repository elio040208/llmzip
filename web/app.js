const inputText = document.querySelector("#inputText");
const totalFreq = document.querySelector("#totalFreq");
const maxContext = document.querySelector("#maxContext");
const traceLimit = document.querySelector("#traceLimit");
const compressBtn = document.querySelector("#compressBtn");
const clearBtn = document.querySelector("#clearBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const copyBase64Btn = document.querySelector("#copyBase64Btn");
const decompressBtn = document.querySelector("#decompressBtn");
const statusBox = document.querySelector("#status");
const metrics = document.querySelector("#metrics");
const hexView = document.querySelector("#hexView");
const base64View = document.querySelector("#base64View");
const decompressedText = document.querySelector("#decompressedText");
const traceTokenCount = document.querySelector("#traceTokenCount");
const tracePayloadBits = document.querySelector("#tracePayloadBits");
const payloadMarker = document.querySelector("#payloadMarker");
const payloadFraction = document.querySelector("#payloadFraction");
const traceRows = document.querySelector("#traceRows");

const sampleText =
  "The quick brown fox jumps over the lazy dog. The quick brown fox jumps again.";

let lastArchiveBase64 = "";

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? "#a42828" : "#607078";
}

function setBusy(isBusy) {
  compressBtn.disabled = isBusy;
  decompressBtn.disabled = isBusy;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  }
  return data;
}

function updateMetrics(result) {
  metrics.innerHTML = `
    <div><span>Input</span><strong>${result.input_bytes ?? "-"} B</strong></div>
    <div><span>Archive</span><strong>${result.archive_bytes ?? "-"} B</strong></div>
  `;
}

function formatUnitInterval(value) {
  if (!Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function formatTokenText(token) {
  const text = token.text || token.piece || String(token.token_id);
  return JSON.stringify(text);
}

function resetTrace() {
  traceTokenCount.textContent = "- tokens";
  tracePayloadBits.textContent = "- bits";
  payloadMarker.style.left = "0%";
  payloadFraction.textContent = "-";
  traceRows.textContent = "";
}

function makeTraceCell(className, text) {
  const element = document.createElement("div");
  element.className = className;
  element.textContent = text;
  return element;
}

function updateTrace(trace) {
  resetTrace();
  if (!trace || !Array.isArray(trace.tokens)) return;

  traceTokenCount.textContent = `${trace.shown_count}/${trace.token_count} tokens`;
  const fraction = trace.payload_fraction || {};
  tracePayloadBits.textContent = `${fraction.bit_count ?? 0} bits`;
  payloadFraction.textContent = `${fraction.binary || "0.0"}  ~= ${fraction.decimal || "0"}`;
  const marker = Math.min(1, Math.max(0, Number(fraction.marker || 0)));
  payloadMarker.style.left = `${marker * 100}%`;

  for (const token of trace.tokens) {
    const row = document.createElement("div");
    row.className = "trace-row";

    row.appendChild(makeTraceCell("trace-index", String(token.index + 1)));

    const tokenCell = document.createElement("div");
    tokenCell.className = "trace-token";
    const tokenCode = document.createElement("code");
    tokenCode.textContent = formatTokenText(token);
    const tokenMeta = document.createElement("span");
    tokenMeta.textContent = `id ${token.token_id}`;
    tokenCell.append(tokenCode, tokenMeta);
    row.appendChild(tokenCell);

    const track = document.createElement("div");
    track.className = "segment-track";
    const fill = document.createElement("div");
    fill.className = "segment-fill";
    const start = Math.min(1, Math.max(0, Number(token.start)));
    const end = Math.min(1, Math.max(start, Number(token.end)));
    const left = start * 100;
    const width = Math.max((end - start) * 100, 0.25);
    fill.style.left = `${left}%`;
    fill.style.width = `${width}%`;
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(makeTraceCell(
      "trace-range",
      `${formatUnitInterval(token.start)}-${formatUnitInterval(token.end)}`,
    ));
    row.appendChild(makeTraceCell(
      "trace-prob",
      `${(Number(token.probability) * 100).toFixed(3)}%`,
    ));
    row.appendChild(makeTraceCell(
      "trace-bits",
      `${Number(token.bits).toFixed(2)} b`,
    ));
    traceRows.appendChild(row);
  }
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  const views = {
    hex: hexView,
    base64: base64View,
  };
  Object.entries(views).forEach(([key, element]) => {
    element.classList.toggle("hidden", key !== name);
  });
}

async function compress() {
  setBusy(true);
  setStatus("Compressing...");
  decompressedText.value = "";
  try {
    const result = await postJson("/api/compress", {
      text: inputText.value,
      total_freq: Number(totalFreq.value),
      max_context: Number(maxContext.value),
      trace_limit: Number(traceLimit.value),
    });
    lastArchiveBase64 = result.archive_base64;
    hexView.textContent = result.archive_hex;
    base64View.value = result.archive_base64;
    updateMetrics(result);
    updateTrace(result.interval_trace);
    showTab("hex");
    setStatus("Compressed");
  } catch (error) {
    setStatus("Error", true);
    hexView.textContent = String(error.message || error);
    resetTrace();
    showTab("hex");
  } finally {
    setBusy(false);
  }
}

async function decompress() {
  const archiveBase64 = base64View.value.trim() || lastArchiveBase64;
  if (!archiveBase64) {
    setStatus("No archive", true);
    return;
  }
  setBusy(true);
  setStatus("Decompressing...");
  try {
    const result = await postJson("/api/decompress", {
      archive_base64: archiveBase64,
    });
    decompressedText.value = result.text;
    setStatus("Decompressed");
  } catch (error) {
    setStatus("Error", true);
    decompressedText.value = String(error.message || error);
  } finally {
    setBusy(false);
  }
}

compressBtn.addEventListener("click", compress);
decompressBtn.addEventListener("click", decompress);

clearBtn.addEventListener("click", () => {
  inputText.value = "";
  decompressedText.value = "";
  hexView.textContent = "";
  base64View.value = "";
  resetTrace();
  lastArchiveBase64 = "";
  setStatus("Ready");
});

sampleBtn.addEventListener("click", () => {
  inputText.value = sampleText;
});

copyBase64Btn.addEventListener("click", async () => {
  const text = base64View.value.trim() || lastArchiveBase64;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  setStatus("Copied");
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});
