/* ============================================================
   Satellite Frame Interpolation — frontend logic
   Uses the current page origin when hosted together, or localhost for file-based preview.
   ============================================================ */

const API_BASE = (() => {
  const configured = window.__APP_CONFIG__?.API_BASE_URL;
  if (configured) return configured;
  if (window.location.protocol === "file:") return "http://localhost:5000";
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "http://localhost:5000";
  }
  return "https://cyan-onions-return.loca.lt";
})();

const state = {
  file0: null,          // File object, when user uploads
  file1: null,
  useSample: false,     // true after "Load sample pair" is clicked
  hasZip: false,        // true once at least one successful run happened
  sweepFrames: [],      // array of {t, image}
};

const el = {
  file0: document.getElementById("file0"),
  file1: document.getElementById("file1"),
  thumb0: document.getElementById("thumb0"),
  thumb1: document.getElementById("thumb1"),
  sampleBtn: document.getElementById("sample-btn"),
  tSlider: document.getElementById("t-slider"),
  tValue: document.getElementById("t-value"),
  regSlider: document.getElementById("reg-slider"),
  regValue: document.getElementById("reg-value"),
  morphToggle: document.getElementById("morph-toggle"),
  morphSlider: document.getElementById("morph-slider"),
  morphPreview: document.getElementById("morph-preview"),
  morphFrame0: document.getElementById("morph-frame0"),
  morphFrame1: document.getElementById("morph-frame1"),
  generateBtn: document.getElementById("generate-btn"),
  sweepBtn: document.getElementById("sweep-btn"),
  clearBtn: document.getElementById("clear-btn"),
  statusMsg: document.getElementById("status-msg"),
  progressRow: document.getElementById("progress-row"),
  progressStep: document.getElementById("progress-step"),
  progressSub: document.getElementById("progress-sub"),
  resThumb0: document.getElementById("res-thumb0"),
  resThumbMid: document.getElementById("res-thumb-mid"),
  resThumb1: document.getElementById("res-thumb1"),
  resT: document.getElementById("res-t"),
  sweepPanel: document.getElementById("sweep-panel"),
  sweepStrip: document.getElementById("sweep-strip"),
  sweepMainImg: document.getElementById("sweep-main-img"),
  sweepSlider: document.getElementById("sweep-slider"),
  compareRow: document.getElementById("compare-row"),
  heatmapPanel: document.getElementById("heatmap-panel"),
  heatmapRow: document.getElementById("heatmap-row"),
  metricsNote: document.getElementById("metrics-note"),
  downloadBtn: document.getElementById("download-btn"),
  backendStatus: document.getElementById("backend-status"),
  lightbox: document.getElementById("lightbox"),
  lightboxImg: document.getElementById("lightbox-img"),
  lightboxClose: document.getElementById("lightbox-close"),
};

const progressSteps = [
  { label: "computing optical flow…", sub: "Estimating motion between the two frames." },
  { label: "warping frames toward t…", sub: "Pre-warping each frame to the requested interpolation time." },
  { label: "refining with OT…", sub: "Running Sinkhorn refinement for the hybrid result." },
];

let loadingTimer = null;
let loadingStepIndex = 0;
let loadingStartedAt = 0;

/* ---------------- helpers ---------------- */

function setThumb(container, src) {
  container.innerHTML = "";
  const img = document.createElement("img");
  img.src = src;
  img.alt = "";
  container.appendChild(img);
}

function clearThumb(container, label = "—") {
  container.innerHTML = `<span class="placeholder">${label}</span>`;
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setStatus(msg, isError = false) {
  el.statusMsg.textContent = msg;
  el.statusMsg.classList.toggle("error", isError);
}

function setMorphPreviewVisible(visible) {
  el.morphPreview.querySelector(".placeholder")?.toggleAttribute("hidden", visible);
  el.morphSlider.disabled = !visible;
}

function syncMorphPreview() {
  if (!el.morphFrame0.src || !el.morphFrame1.src) return;
  const alpha = Number(el.morphSlider.value);
  el.morphFrame0.style.opacity = String(1 - alpha);
  el.morphFrame1.style.opacity = String(alpha);
}

function openLightbox(src) {
  el.lightboxImg.src = src;
  el.lightbox.hidden = false;
}

function closeLightbox() {
  el.lightbox.hidden = true;
  el.lightboxImg.removeAttribute("src");
}

function setLoadingUI(isLoading) {
  el.progressRow.hidden = !isLoading;
  const loadingThumbs = document.querySelectorAll("#result-panel .thumb, #compare-row .thumb");
  loadingThumbs.forEach((thumb) => {
    thumb.classList.toggle("is-loading", isLoading);
    if (isLoading && !thumb.querySelector("img")) {
      thumb.innerHTML = '<span class="placeholder">processing…</span>';
    }
  });
}

function updateLoadingUI(elapsedSeconds = 0) {
  const step = progressSteps[loadingStepIndex] || progressSteps[progressSteps.length - 1];
  el.progressStep.textContent = `${step.label}${elapsedSeconds ? ` · ${elapsedSeconds}s` : ""}`;
  el.progressSub.textContent = step.sub;
}

function startLoadingUI() {
  clearInterval(loadingTimer);
  loadingStepIndex = 0;
  loadingStartedAt = performance.now();
  setLoadingUI(true);
  updateLoadingUI();

  loadingTimer = window.setInterval(() => {
    const elapsedSeconds = ((performance.now() - loadingStartedAt) / 1000).toFixed(1);
    const stepIndex = Math.min(progressSteps.length - 1, Math.floor(Number(elapsedSeconds) / 1.4));
    loadingStepIndex = stepIndex;
    updateLoadingUI(elapsedSeconds);
  }, 300);
}

function stopLoadingUI() {
  clearInterval(loadingTimer);
  setLoadingUI(false);
  el.progressStep.textContent = "ready";
  el.progressSub.textContent = "The hybrid pipeline runs optical flow and OT refinement.";
}

function canGenerate() {
  return state.useSample || (state.file0 && state.file1);
}

function refreshGenerateButton() {
  el.generateBtn.disabled = !canGenerate();
}

/* ---------------- backend health check ---------------- */

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      el.backendStatus.textContent = "backend online";
      el.backendStatus.className = "mono ok";
      return;
    }
    throw new Error("bad response");
  } catch (e) {
    el.backendStatus.textContent = "backend offline — start the Flask server on :5000";
    el.backendStatus.className = "mono down";
  }
}

/* ---------------- upload handling ---------------- */

async function handleFileChange(which) {
  const input = which === 0 ? el.file0 : el.file1;
  const thumb = which === 0 ? el.thumb0 : el.thumb1;
  const file = input.files && input.files[0];
  if (!file) return;

  state.useSample = false;
  if (which === 0) state.file0 = file;
  else state.file1 = file;

  const dataUrl = await fileToDataURL(file);
  setThumb(thumb, dataUrl);
  if (which === 0) {
    el.morphFrame0.src = dataUrl;
    el.morphFrame0.hidden = false;
  } else {
    el.morphFrame1.src = dataUrl;
    el.morphFrame1.hidden = false;
  }
  setMorphPreviewVisible(Boolean(el.morphFrame0.src && el.morphFrame1.src));
  syncMorphPreview();
  refreshGenerateButton();
}

el.file0.addEventListener("change", () => handleFileChange(0));
el.file1.addEventListener("change", () => handleFileChange(1));

document.querySelectorAll("[data-target]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById(btn.dataset.target).click();
  });
});

/* ---------------- sample pair ---------------- */

el.sampleBtn.addEventListener("click", async () => {
  setStatus("loading sample pair…");
  try {
    const res = await fetch(`${API_BASE}/sample-pair`);
    if (!res.ok) throw new Error("failed to fetch sample pair");
    const data = await res.json();

    state.useSample = true;
    state.file0 = null;
    state.file1 = null;
    el.file0.value = "";
    el.file1.value = "";

    setThumb(el.thumb0, data.frame0);
    setThumb(el.thumb1, data.frame1);
    el.morphFrame0.src = data.frame0;
    el.morphFrame1.src = data.frame1;
    el.morphFrame0.hidden = false;
    el.morphFrame1.hidden = false;
    setMorphPreviewVisible(true);
    syncMorphPreview();
    refreshGenerateButton();
    setStatus("sample pair loaded.");
  } catch (e) {
    setStatus(`could not load sample pair (${e.message})`, true);
  }
});

/* ---------------- slider ---------------- */

el.tSlider.addEventListener("input", () => {
  const t = parseFloat(el.tSlider.value).toFixed(2);
  el.tValue.textContent = t;
});

el.regSlider.addEventListener("input", () => {
  const reg = parseFloat(el.regSlider.value).toFixed(3);
  el.regValue.textContent = reg;
});

el.morphToggle.addEventListener("click", () => {
  const visible = !el.morphPreview.classList.contains("is-active");
  el.morphPreview.classList.toggle("is-active", visible);
  el.morphToggle.textContent = visible ? "hide crossfade" : "show crossfade";
  el.morphFrame0.hidden = !visible;
  el.morphFrame1.hidden = !visible;
  setMorphPreviewVisible(visible && Boolean(el.morphFrame0.src && el.morphFrame1.src));
  syncMorphPreview();
});

el.morphSlider.addEventListener("input", syncMorphPreview);

/* ---------------- generate ---------------- */

async function runInterpolation(t, reg = Number(el.regSlider.value)) {
  const form = new FormData();
  form.append("t", String(t));
  form.append("reg", String(reg));

  if (state.useSample) {
    form.append("is_sample", "1");
  } else {
    form.append("frame0", state.file0);
    form.append("frame1", state.file1);
  }

  const res = await fetch(`${API_BASE}/interpolate`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error || `server responded ${res.status}`);
  }

  return res.json();
}

el.generateBtn.addEventListener("click", async () => {
  if (!canGenerate()) return;

  const t = parseFloat(el.tSlider.value);
  el.generateBtn.disabled = true;
  setStatus("running interpolation…");
  startLoadingUI();

  try {
    const data = await runInterpolation(t);
    renderResult(data);
    setStatus(`done — t = ${data.t.toFixed(2)}`);
    state.hasZip = true;
    el.downloadBtn.disabled = false;
  } catch (e) {
    setStatus(`interpolation failed (${e.message})`, true);
  } finally {
    stopLoadingUI();
    refreshGenerateButton();
  }
});

el.sweepBtn.addEventListener("click", async () => {
  if (!canGenerate()) return;

  el.generateBtn.disabled = true;
  el.sweepBtn.disabled = true;
  setStatus("building full sweep…");
  startLoadingUI();

  try {
    const frames = [];
    for (let step = 0; step <= 10; step += 1) {
      const t = step / 10;
      const data = await runInterpolation(t);
      frames.push({ t, image: data.interpolated });
    }

    state.sweepFrames = frames;
    renderSweep(frames);
    setStatus("full sweep ready");
  } catch (e) {
    setStatus(`sweep failed (${e.message})`, true);
  } finally {
    stopLoadingUI();
    el.generateBtn.disabled = false;
    el.sweepBtn.disabled = false;
  }
});

el.clearBtn.addEventListener("click", () => {
  state.file0 = null;
  state.file1 = null;
  state.useSample = false;
  state.sweepFrames = [];
  el.file0.value = "";
  el.file1.value = "";
  clearThumb(el.thumb0, "no frame");
  clearThumb(el.thumb1, "no frame");
  clearThumb(el.resThumb0, "—");
  clearThumb(el.resThumbMid, "—");
  clearThumb(el.resThumb1, "—");
  el.resT.textContent = "t=0.50";
  el.sweepPanel.hidden = true;
  el.sweepStrip.innerHTML = "";
  el.sweepMainImg.hidden = true;
  el.sweepMainImg.removeAttribute("src");
  el.sweepSlider.disabled = true;
  el.sweepSlider.value = "0";
  el.downloadBtn.disabled = true;
  setStatus("cleared");
  refreshGenerateButton();
});

/* ---------------- render results ---------------- */

const METHOD_LABELS = {
  linear_blend: "linear blend",
  flow_only_warp: "optical-flow warp",
  hybrid_ot: "ot-hybrid",
};

function renderSweep(frames) {
  el.sweepPanel.hidden = false;
  el.sweepStrip.innerHTML = "";
  el.sweepMainImg.hidden = false;

  frames.forEach((frame, index) => {
    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = "sweep-thumb";
    thumb.dataset.index = String(index);
    const img = document.createElement("img");
    img.src = frame.image;
    img.alt = `t=${frame.t.toFixed(1)}`;
    thumb.appendChild(img);
    thumb.addEventListener("click", () => {
      el.sweepSlider.value = String(index);
      el.sweepMainImg.src = frame.image;
    });
    el.sweepStrip.appendChild(thumb);
  });

  if (frames.length) {
    el.sweepMainImg.src = frames[0].image;
    el.sweepSlider.disabled = false;
    el.sweepSlider.max = String(Math.max(0, frames.length - 1));
    el.sweepSlider.value = "0";
  }
}

el.sweepSlider.addEventListener("input", () => {
  const index = Number(el.sweepSlider.value);
  const frame = state.sweepFrames[index];
  if (frame) {
    el.sweepMainImg.src = frame.image;
  }
});

function renderResult(data) {
  // Top triplet: frame0 | interpolated | frame1
  const src0 = state.useSample ? el.thumb0.querySelector("img")?.src : null;
  const src1 = state.useSample ? el.thumb1.querySelector("img")?.src : null;

  // Prefer whatever is already shown in the upload thumbnails (identical source
  // images), falling back to re-reading uploaded files if needed.
  const thumb0Img = el.thumb0.querySelector("img");
  const thumb1Img = el.thumb1.querySelector("img");
  if (thumb0Img) setThumb(el.resThumb0, thumb0Img.src);
  if (thumb1Img) setThumb(el.resThumb1, thumb1Img.src);

  setThumb(el.resThumbMid, data.interpolated);
  el.resT.textContent = `t=${data.t.toFixed(2)}`;

  // Heatmap strip (difference vs ground truth)
  el.heatmapPanel.hidden = !data.has_metrics || !data.ground_truth;
  if (data.has_metrics && data.ground_truth) {
    const cards = [];
    Object.entries(data.methods || {}).forEach(([key, entry]) => {
      if (!entry || !entry.heatmap) return;
      const label = METHOD_LABELS[key] || key;
      const wrapper = document.createElement("figure");
      wrapper.className = "heatmap-card";
      wrapper.innerHTML = `
        <div class="thumb"><span class="placeholder">heatmap</span></div>
        <figcaption>
          <span class="heatmap-label">${label}</span>
          <span class="heatmap-meta">abs difference vs. ground truth</span>
        </figcaption>
      `;
      const thumb = wrapper.querySelector(".thumb");
      const img = document.createElement("img");
      img.src = entry.heatmap;
      img.alt = `${label} heatmap`;
      thumb.appendChild(img);
      cards.push(wrapper);
    });

    el.heatmapRow.innerHTML = "";
    cards.forEach((card) => el.heatmapRow.appendChild(card));
  } else {
    el.heatmapRow.innerHTML = "";
  }

  // Comparison strip
  const cards = el.compareRow.querySelectorAll(".method-card");
  cards.forEach((card) => {
    const key = card.dataset.method;
    const entry = data.methods && data.methods[key];
    const thumb = card.querySelector(".thumb");
    const metricsSpan = card.querySelector(".method-metrics");
    const hintSpan = card.querySelector(".method-hint");

    if (entry) {
      setThumb(thumb, entry.image);
      if (data.has_metrics && entry.psnr !== undefined) {
        const ssimValue = Number(entry.ssim).toFixed(4);
        const psnrValue = Number(entry.psnr).toFixed(2);
        metricsSpan.innerHTML = `PSNR <span class="mono">${psnrValue} dB</span> · <strong>SSIM ${ssimValue}</strong>`;
        hintSpan.textContent = "higher SSIM means the structure is closer to the ground truth";
      } else {
        metricsSpan.textContent = "metrics unavailable";
        hintSpan.textContent = "ground-truth metrics appear here when available";
      }
    } else {
      clearThumb(thumb);
      metricsSpan.textContent = "—";
      hintSpan.textContent = "ground-truth metrics appear here when available";
    }
  });

  el.metricsNote.textContent = data.has_metrics
    ? "For this demo, SSIM is emphasized because it better reflects structural preservation than a raw pixel average."
    : "Upload the sample pair or your own ground-truth frame to see PSNR and SSIM for each method.";
}

/* ---------------- download all ---------------- */

el.downloadBtn.addEventListener("click", async () => {
  el.downloadBtn.disabled = true;
  const original = el.downloadBtn.textContent;
  el.downloadBtn.textContent = "preparing…";
  try {
    const res = await fetch(`${API_BASE}/download-zip`);
    if (!res.ok) throw new Error(`server responded ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sat-frame-interp-results.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    setStatus(`download failed (${e.message})`, true);
  } finally {
    el.downloadBtn.textContent = original;
    el.downloadBtn.disabled = false;
  }
});

/* ---------------- lightbox ---------------- */

el.lightboxClose.addEventListener("click", closeLightbox);
el.lightbox.addEventListener("click", (e) => {
  if (e.target === el.lightbox) closeLightbox();
});

document.addEventListener("click", (event) => {
  const thumb = event.target.closest(".thumb");
  if (!thumb) return;
  const img = thumb.querySelector("img");
  if (img && img.src) openLightbox(img.src);
});

/* ---------------- init ---------------- */

refreshGenerateButton();
checkBackend();
