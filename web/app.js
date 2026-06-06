const inputText = document.querySelector("#inputText");
const totalFreq = document.querySelector("#totalFreq");
const maxContext = document.querySelector("#maxContext");
const compressBtn = document.querySelector("#compressBtn");
const clearBtn = document.querySelector("#clearBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const copyBase64Btn = document.querySelector("#copyBase64Btn");
const decompressBtn = document.querySelector("#decompressBtn");
const statusBox = document.querySelector("#status");
const metrics = document.querySelector("#metrics");
const hexView = document.querySelector("#hexView");
const base64View = document.querySelector("#base64View");
const headerView = document.querySelector("#headerView");
const decompressedText = document.querySelector("#decompressedText");

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

function formatNumber(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
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
  const meta = result.metadata || {};
  metrics.innerHTML = `
    <div><span>Input</span><strong>${result.input_bytes ?? "-"} B</strong></div>
    <div><span>Archive</span><strong>${result.archive_bytes ?? "-"} B</strong></div>
    <div><span>Payload BPB</span><strong>${formatNumber(meta.bpb_payload, 3)}</strong></div>
    <div><span>File BPB</span><strong>${formatNumber(meta.bpb_file, 3)}</strong></div>
  `;
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  const views = {
    hex: hexView,
    base64: base64View,
    header: headerView,
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
    });
    lastArchiveBase64 = result.archive_base64;
    hexView.textContent = result.archive_hex;
    base64View.value = result.archive_base64;
    headerView.textContent = JSON.stringify(
      {
        parsed_header: result.parsed_header,
        metadata: result.metadata,
      },
      null,
      2,
    );
    updateMetrics(result);
    showTab("hex");
    setStatus("Compressed");
  } catch (error) {
    setStatus("Error", true);
    headerView.textContent = String(error.message || error);
    showTab("header");
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
  headerView.textContent = "";
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
