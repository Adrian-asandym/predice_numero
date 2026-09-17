"use strict";

const pad = document.getElementById("pad");
const preview = document.getElementById("preview");
const bigDigit = document.getElementById("big-digit");
const confidence = document.getElementById("confidence");
const barsList = document.getElementById("bars");
const statusEl = document.getElementById("status");
const clearBtn = document.getElementById("clear-btn");

const WORK = 280;
const offscreen = document.createElement("canvas");
offscreen.width = WORK;
offscreen.height = WORK;
const offCtx = offscreen.getContext("2d", { willReadFrequently: true });

const previewCtx = preview.getContext("2d");
const previewImage = previewCtx.createImageData(28, 28);

let weights = null;
let drawing = false;
let hasInk = false;
let lastPredict = 0;
const padCtx = pad.getContext("2d");

const barRows = [];
for (let i = 0; i < 10; i++) {
  const row = document.createElement("li");
  row.className = "bar-row";
  const label = document.createElement("span");
  label.className = "bar-label";
  label.textContent = i;
  const track = document.createElement("div");
  track.className = "bar-track";
  const fill = document.createElement("div");
  fill.className = "bar-fill";
  track.appendChild(fill);
  const pct = document.createElement("span");
  pct.className = "bar-pct";
  pct.textContent = "0.0%";
  row.appendChild(label);
  row.appendChild(track);
  row.appendChild(pct);
  barsList.appendChild(row);
  barRows.push({ row, fill, pct });
}

function setupCanvas() {
  const rect = pad.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  pad.width = Math.round(rect.width * dpr);
  pad.height = Math.round(rect.width * dpr);
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.strokeStyle = "#ffffff";
  padCtx.fillStyle = "#ffffff";
  padCtx.lineWidth = Math.max(10, pad.width * 0.06);
  clearPad();
}

function clearPad() {
  padCtx.clearRect(0, 0, pad.width, pad.height);
  hasInk = false;
  resetSidebar();
}

function resetSidebar() {
  bigDigit.textContent = "—";
  confidence.textContent = weights ? "Dibuja un número para empezar" : "Cargando modelo…";
  for (const { row, fill, pct } of barRows) {
    row.classList.remove("top");
    fill.style.width = "0%";
    pct.textContent = "0.0%";
  }
  previewCtx.clearRect(0, 0, 28, 28);
}

function pointerPos(e) {
  const rect = pad.getBoundingClientRect();
  const scale = pad.width / rect.width;
  return {
    x: (e.clientX - rect.left) * scale,
    y: (e.clientY - rect.top) * scale,
  };
}

pad.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  const { x, y } = pointerPos(e);
  padCtx.beginPath();
  padCtx.moveTo(x, y);
  padCtx.lineTo(x + 0.01, y + 0.01);
  padCtx.stroke();
  hasInk = true;
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = pointerPos(e);
  padCtx.lineTo(x, y);
  padCtx.stroke();
  const now = performance.now();
  if (now - lastPredict > 150) {
    lastPredict = now;
    predictNow();
  }
});

function endStroke() {
  if (!drawing) return;
  drawing = false;
  predictNow();
}

pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);
clearBtn.addEventListener("click", clearPad);

function renderPreview(input) {
  const data = previewImage.data;
  for (let i = 0; i < 784; i++) {
    const v = Math.round(input[i] * 255);
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  previewCtx.putImageData(previewImage, 0, 0);
}

function predictNow() {
  if (!weights || !hasInk) return;
  offCtx.drawImage(pad, 0, 0, WORK, WORK);
  const img = offCtx.getImageData(0, 0, WORK, WORK);
  const input = DigitModel.preprocess(img.data, WORK, 28);
  if (!input) {
    resetSidebar();
    return;
  }
  const probs = DigitModel.predict(weights, input);
  let top = 0;
  for (let i = 1; i < 10; i++) if (probs[i] > probs[top]) top = i;
  bigDigit.textContent = top;
  confidence.textContent = (probs[top] * 100).toFixed(1) + "% de confianza";
  for (let i = 0; i < 10; i++) {
    const p = probs[i] * 100;
    const { row, fill, pct } = barRows[i];
    row.classList.toggle("top", i === top);
    fill.style.width = p.toFixed(1) + "%";
    pct.textContent = p.toFixed(1) + "%";
  }
  renderPreview(input);
}

function decodeBase64(b64) {
  const binStr = atob(b64);
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
  return bytes.buffer;
}

async function loadWeights() {
  try {
    const resp = await fetch("group1-shard1of1.bin");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return await resp.arrayBuffer();
  } catch (err) {
    if (typeof WEIGHTS_BASE64 === "string" && WEIGHTS_BASE64.length > 0) {
      return decodeBase64(WEIGHTS_BASE64);
    }
    throw err;
  }
}

(async () => {
  setupCanvas();
  try {
    const buffer = await loadWeights();
    weights = DigitModel.parseWeights(buffer);
    statusEl.textContent = "Modelo listo";
    statusEl.classList.add("ok");
    resetSidebar();
  } catch (err) {
    statusEl.textContent = "Error al cargar los pesos del modelo";
    statusEl.classList.add("error");
    confidence.textContent = "No se pudo cargar el modelo";
    console.error(err);
  }
})();
