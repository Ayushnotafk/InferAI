// -------------------------------------------------------------
// Interactive State Management
// -------------------------------------------------------------
const DEMO_EXAMPLES = {
  observation: "On the telemetry screen, the pressure trace flatlined for six seconds after the valve command. The operator log records the same gap on two consecutive shifts.",
  inference: "Latency spikes only during failover in this service, which suggests a race in the controller path. Therefore the next debugging step should focus on shared mutable state.",
  analogy: "Teaching students about attention in transformers is similar to explaining a spotlight operator: the model learns where to look next based on what it already took in.",
  authority: "According to the published infection-control guideline, hand hygiene remains a cornerstone of prevention. Experts in the field treat that recommendation as a minimum bar for compliance audits."
};

// UI Elements
const apiInput = document.getElementById("api-url-input");
const alphaSlider = document.getElementById("alpha-slider");
const alphaValLabel = document.getElementById("alpha-val");
const shapCheckbox = document.getElementById("shap-checkbox");
const argumentTextarea = document.getElementById("argument-textarea");
const analyzeBtn = document.getElementById("analyze-btn");
const statusMsg = document.getElementById("status-message");
const resultsPanel = document.getElementById("results-panel");
const resultsContent = document.querySelector(".results-content");
const placeholderView = document.querySelector(".results-placeholder");

// Results Elements
const finalLabelSpotlight = document.getElementById("final-label-spotlight");
const mlLabelSub = document.getElementById("ml-label-sub");
const hybridConfVal = document.getElementById("hybrid-conf-val");
const hybridConfBar = document.getElementById("hybrid-conf-bar");
const mlConfVal = document.getElementById("ml-conf-val");
const mlConfBar = document.getElementById("ml-conf-bar");
const strengthBadgeWrapper = document.getElementById("strength-badge-wrapper");
const extractedClaim = document.getElementById("extracted-claim");
const extractedPremises = document.getElementById("extracted-premises");
const signalsList = document.getElementById("signals-list");
const highlightedCuesBox = document.getElementById("highlighted-cues-box");
const explanationText = document.getElementById("explanation-text");
const shapSection = document.getElementById("shap-section");
const shapNote = document.getElementById("shap-note");
const shapSvgChart = document.getElementById("shap-svg-chart");
const chartTableBody = document.getElementById("chart-table-body");

let lastAnalyzedText = "";
let debounceTimer = null;

// Initialize Settings from LocalStorage if available
if (localStorage.getItem("inferai_api_url")) {
  apiInput.value = localStorage.getItem("inferai_api_url");
}
if (localStorage.getItem("inferai_include_shap")) {
  shapCheckbox.checked = localStorage.getItem("inferai_include_shap") === "true";
}

// -------------------------------------------------------------
// Event Listeners
// -------------------------------------------------------------

// Slider Real-time Value Updates & Auto-retrigger
alphaSlider.addEventListener("input", (e) => {
  const val = parseFloat(e.target.value).toFixed(2);
  alphaValLabel.textContent = val;
  
  // Real-time slider re-runs analysis on the same text after small debounce
  if (lastAnalyzedText) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      runDiagnostics(true); // run silently/softly
    }, 150);
  }
});

// Examples Buttons
document.querySelectorAll(".btn-example").forEach(btn => {
  btn.addEventListener("click", (e) => {
    // Toggle active state
    document.querySelectorAll(".btn-example").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    
    // Set text and trigger
    const type = e.target.getAttribute("data-type");
    argumentTextarea.value = DEMO_EXAMPLES[type] || "";
    runDiagnostics();
  });
});

// Run Button
analyzeBtn.addEventListener("click", () => {
  runDiagnostics();
});

// Save settings on input change
apiInput.addEventListener("change", () => {
  localStorage.setItem("inferai_api_url", apiInput.value.trim());
});
shapCheckbox.addEventListener("change", () => {
  localStorage.setItem("inferai_include_shap", shapCheckbox.checked);
  if (lastAnalyzedText) runDiagnostics();
});

// -------------------------------------------------------------
// Core Actions & API Integration
// -------------------------------------------------------------

async function runDiagnostics(silent = false) {
  const text = argumentTextarea.value.trim();
  if (!text) {
    showStatus("Please enter an argument to analyze.", "error");
    return;
  }

  const baseApiUrl = apiInput.value.trim() || "http://127.0.0.1:8000";
  const alpha = parseFloat(alphaSlider.value);
  const includeShap = shapCheckbox.checked;

  lastAnalyzedText = text;

  if (!silent) {
    showStatus(`<span class="nyx-loader"></span> Fetching Neuro-Symbolic blend from ${baseApiUrl}...`, "loading");
    analyzeBtn.disabled = true;
  }

  try {
    const response = await fetch(`${baseApiUrl.replace(/\/$/, "")}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text,
        include_shap: includeShap,
        alpha: alpha
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    renderResults(data);
    
    if (!silent) {
      showStatus("Diagnostics computed successfully.", "success");
      setTimeout(() => statusMsg.classList.add("hide"), 3000);
    }
  } catch (err) {
    console.error(err);
    if (!silent) {
      showStatus(`Could not reach the analysis service at ${baseApiUrl}. Make sure the FastAPI server is running. Error: ${err.message}`, "error");
    }
  } finally {
    if (!silent) {
      analyzeBtn.disabled = false;
    }
  }
}

// Status Display Helper
function showStatus(msg, type) {
  statusMsg.innerHTML = msg;
  statusMsg.className = "status-msg " + type;
}

// -------------------------------------------------------------
// Rendering Results & Visual Charts
// -------------------------------------------------------------

function renderResults(data) {
  // Reveal content panel, hide placeholder
  resultsPanel.classList.remove("placeholder-active");
  placeholderView.classList.add("hide");
  resultsContent.classList.remove("hide");

  const hybrid = data.hybrid || {};
  const finalLabel = data.hybrid_predicted_pramana || data.predicted_pramana || "—";
  const mlLabel = data.predicted_pramana || "—";
  const hybridConf = parseConfidence(data.adjusted_confidence);
  const mlConf = parseConfidence(data.confidence);
  const strength = data.reasoning_strength || "—";

  // Spotlight & Badges
  finalLabelSpotlight.textContent = `✦ ${finalLabel}`;
  mlLabelSub.textContent = `Reference ML label: ${mlLabel}`;
  
  hybridConfVal.textContent = `${hybridConf.toFixed(0)}%`;
  hybridConfBar.style.width = `${hybridConf}%`;
  
  mlConfVal.textContent = `${mlConf.toFixed(0)}%`;
  mlConfBar.style.width = `${mlConf}%`;

  renderStrengthBadge(strength);

  // Extracted Structure
  extractedClaim.textContent = data.claim || "—";
  
  // Format premises (could be list or string)
  if (Array.isArray(data.premises)) {
    extractedPremises.textContent = data.premises.join("\n\n") || "—";
  } else {
    extractedPremises.textContent = data.premises || data.evidence || "—";
  }

  // Detected cues highlight
  highlightedCuesBox.innerHTML = data.highlighted_html || `<p>${escapeHTML(data.input_text)}</p>`;

  // Explanatory Text
  explanationText.textContent = data.explanation || "—";

  // Heuristic Signals Badges
  signalsList.innerHTML = "";
  const signals = hybrid.pattern_signals || {};
  const signalLabels = {
    observation_hits: "👁️ Observation hits",
    inference_hits: "⛓️ Inference hits",
    analogy_hits: "⚖️ Analogy hits",
    authority_hits: "📖 Authority hits"
  };
  let signalCount = 0;
  for (const [key, value] of Object.entries(signals)) {
    if (signalLabels[key]) {
      const badge = document.createElement("div");
      badge.className = "signal-badge";
      badge.innerHTML = `${signalLabels[key]}: <span>${value}</span>`;
      signalsList.appendChild(badge);
      signalCount += value;
    }
  }
  if (signalCount === 0) {
    signalsList.innerHTML = `<span style="font-size:0.85rem;color:var(--text-dim);font-style:italic;">No symbolic patterns detected in current argument. Rules will output flat uniform distribution.</span>`;
  }

  // Probability Blending Chart & Table
  renderBlendingChart(hybrid);

  // SHAP Contributions (If present)
  if (data.shap && !data.shap.error) {
    shapSection.classList.remove("hide");
    shapNote.textContent = data.shap.note || "SHAP value summary over embeddings.";
    renderShapChart(data.shap);
  } else {
    shapSection.classList.add("hide");
  }
}

// Render the reasoning strength badge with appropriate colors
function renderStrengthBadge(strength) {
  strengthBadgeWrapper.innerHTML = "";
  const cleanStr = strength.trim().toLowerCase();
  
  const badge = document.createElement("span");
  badge.className = "nyx-badge";
  
  let emoji = "○";
  if (cleanStr === "strong" || cleanStr === "high") {
    badge.classList.add("nyx-badge-strong");
    emoji = "◆";
  } else if (cleanStr === "moderate" || cleanStr === "medium") {
    badge.classList.add("nyx-badge-moderate");
    emoji = "◇";
  } else {
    badge.classList.add("nyx-badge-weak");
  }
  
  badge.textContent = `${emoji} ${strength}`;
  strengthBadgeWrapper.appendChild(badge);
}

// -------------------------------------------------------------
// SVG Visualization Drawing Logic
// -------------------------------------------------------------

function renderBlendingChart(hybrid) {
  const svg = document.getElementById("blending-svg-chart");
  svg.innerHTML = ""; // Clear SVG

  const classes = hybrid.class_order || ["Anumana", "Pratyaksha", "Shabda", "Upamana"];
  const mlProbs = (hybrid.ml_probs || []).map(p => p * 100);
  const ruleProbs = (hybrid.rule_probs || []).map(p => p * 100);
  const fusedProbs = (hybrid.fused_probs || []).map(p => p * 100);

  // Layout params
  const width = 500;
  const height = 240;
  const margin = { top: 15, right: 30, bottom: 25, left: 90 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Grid lines (vertical ticks)
  const ticks = [0, 25, 50, 75, 100];
  ticks.forEach(t => {
    const x = margin.left + (t / 100) * chartWidth;
    
    // Vertical line
    const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gridLine.setAttribute("x1", x);
    gridLine.setAttribute("y1", margin.top);
    gridLine.setAttribute("x2", x);
    gridLine.setAttribute("y2", margin.top + chartHeight);
    gridLine.setAttribute("stroke", "rgba(16, 185, 129, 0.08)");
    gridLine.setAttribute("stroke-dasharray", "3,3");
    svg.appendChild(gridLine);

    // Text labels at bottom
    const gridLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    gridLabel.setAttribute("x", x);
    gridLabel.setAttribute("y", margin.top + chartHeight + 15);
    gridLabel.setAttribute("font-family", "var(--font-family-sans)");
    gridLabel.setAttribute("font-size", "9px");
    gridLabel.setAttribute("fill", "var(--text-dim)");
    gridLabel.setAttribute("text-anchor", "middle");
    gridLabel.textContent = `${t}%`;
    svg.appendChild(gridLabel);
  });

  // Render bars for each category
  const numClasses = classes.length;
  const groupHeight = chartHeight / numClasses;
  const barHeight = 8;
  const barGap = 3;

  // Clear and rebuild table
  chartTableBody.innerHTML = "";

  classes.forEach((className, i) => {
    const groupCenterY = margin.top + (i * groupHeight) + (groupHeight / 2);
    
    // Class name label
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", margin.left - 12);
    label.setAttribute("y", groupCenterY + 4);
    label.setAttribute("font-family", "var(--font-family-sans)");
    label.setAttribute("font-size", "11px");
    label.setAttribute("font-weight", "600");
    label.setAttribute("fill", "var(--text-main)");
    label.setAttribute("text-anchor", "end");
    label.textContent = className;
    svg.appendChild(label);

    // Draw three bars (ML, Rule, Hybrid)
    const values = [
      { pct: mlProbs[i] || 0, color: "#10b981", yOffset: -barHeight - barGap },
      { pct: ruleProbs[i] || 0, color: "#f59e0b", yOffset: -barHeight / 2 },
      { pct: fusedProbs[i] || 0, color: "#0ea5e9", yOffset: barGap + barHeight / 2 }
    ];

    values.forEach(barInfo => {
      const barWidth = Math.max(2, (barInfo.pct / 100) * chartWidth);
      
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", margin.left);
      rect.setAttribute("y", groupCenterY + barInfo.yOffset);
      rect.setAttribute("width", 0); // Start at 0 for animation
      rect.setAttribute("height", barHeight);
      rect.setAttribute("rx", "3");
      rect.setAttribute("fill", barInfo.color);
      rect.style.transition = "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)";
      svg.appendChild(rect);

      // Trigger width animation in next frame
      requestAnimationFrame(() => {
        rect.setAttribute("width", barWidth);
      });
    });

    // Add table row
    const tr = document.createElement("tr");
    
    const tdClass = document.createElement("td");
    tdClass.className = "table-cell-bold";
    tdClass.textContent = className;
    tr.appendChild(tdClass);

    const tdMl = document.createElement("td");
    tdMl.className = "text-right";
    tdMl.textContent = `${(mlProbs[i] || 0).toFixed(1)}%`;
    tr.appendChild(tdMl);

    const tdRules = document.createElement("td");
    tdRules.className = "text-right";
    tdRules.textContent = `${(ruleProbs[i] || 0).toFixed(1)}%`;
    tr.appendChild(tdRules);

    const tdHybrid = document.createElement("td");
    tdHybrid.className = "text-right table-cell-highlight";
    tdHybrid.textContent = `${(fusedProbs[i] || 0).toFixed(1)}%`;
    tr.appendChild(tdHybrid);

    chartTableBody.appendChild(tr);
  });
}

function renderShapChart(shap) {
  const svg = document.getElementById("shap-svg-chart");
  svg.innerHTML = ""; // Clear SVG

  const contributions = shap.top_embedding_contributions || [];
  if (contributions.length === 0) return;

  const width = 500;
  const height = 300;
  const margin = { top: 15, right: 20, bottom: 25, left: 100 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Find max absolute SHAP value for scaling
  let maxAbsVal = 0.001;
  contributions.forEach(d => {
    const val = Math.abs(d.shap_value || 0);
    if (val > maxAbsVal) maxAbsVal = val;
  });

  const centerLineX = margin.left + (chartWidth / 2);

  // Vertical axis center line
  const centerLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  centerLine.setAttribute("x1", centerLineX);
  centerLine.setAttribute("y1", margin.top);
  centerLine.setAttribute("x2", centerLineX);
  centerLine.setAttribute("y2", margin.top + chartHeight);
  centerLine.setAttribute("stroke", "rgba(16, 185, 129, 0.3)");
  centerLine.setAttribute("stroke-width", "1.5");
  svg.appendChild(centerLine);

  // Grid lines
  const tickVals = [-maxAbsVal, -maxAbsVal / 2, 0, maxAbsVal / 2, maxAbsVal];
  tickVals.forEach(v => {
    const x = centerLineX + (v / maxAbsVal) * (chartWidth / 2);
    
    if (v !== 0) {
      const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
      grid.setAttribute("x1", x);
      grid.setAttribute("y1", margin.top);
      grid.setAttribute("x2", x);
      grid.setAttribute("y2", margin.top + chartHeight);
      grid.setAttribute("stroke", "rgba(16, 185, 129, 0.05)");
      grid.setAttribute("stroke-dasharray", "2,2");
      svg.appendChild(grid);
    }

    // Grid labels
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x);
    label.setAttribute("y", margin.top + chartHeight + 15);
    label.setAttribute("font-family", "var(--font-family-mono)");
    label.setAttribute("font-size", "8px");
    label.setAttribute("fill", "var(--text-dim)");
    label.setAttribute("text-anchor", "middle");
    label.textContent = v.toFixed(3);
    svg.appendChild(label);
  });

  // Render SHAP Bars
  const numBars = contributions.length;
  const barSpacing = chartHeight / numBars;
  const barHeight = Math.min(14, barSpacing - 6);

  contributions.forEach((item, i) => {
    const y = margin.top + (i * barSpacing) + (barSpacing - barHeight) / 2;
    const shapVal = item.shap_value || 0;
    
    // Label: Dimension name
    const textLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    textLabel.setAttribute("x", margin.left - 10);
    textLabel.setAttribute("y", y + (barHeight / 2) + 4);
    textLabel.setAttribute("font-family", "var(--font-family-mono)");
    textLabel.setAttribute("font-size", "9px");
    textLabel.setAttribute("fill", "var(--text-muted)");
    textLabel.setAttribute("text-anchor", "end");
    textLabel.textContent = `Dim ${item.embedding_dim}`;
    svg.appendChild(textLabel);

    // Compute width and start X based on positive vs negative
    const barWidth = Math.max(1, (Math.abs(shapVal) / maxAbsVal) * (chartWidth / 2));
    const startX = shapVal >= 0 ? centerLineX : centerLineX - barWidth;
    const barColor = shapVal >= 0 ? "#10b981" : "#ef4444"; // Green for positive, Red for negative

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", centerLineX); // Start animation from center
    rect.setAttribute("y", y);
    rect.setAttribute("width", 0);
    rect.setAttribute("height", barHeight);
    rect.setAttribute("rx", "2");
    rect.setAttribute("fill", barColor);
    rect.style.transition = "x 0.8s ease-out, width 0.8s ease-out";
    svg.appendChild(rect);

    requestAnimationFrame(() => {
      rect.setAttribute("x", startX);
      rect.setAttribute("width", barWidth);
    });
  });
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------

function parseConfidence(val) {
  if (val === undefined || val === null) return 0;
  // If it's a string like "82.5%", strip percentage
  if (typeof val === "string") {
    val = val.replace("%", "").trim();
  }
  const parsed = parseFloat(val);
  return isNaN(parsed) ? 0 : Math.max(0, Math.min(100, parsed));
}

function escapeHTML(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, 
    tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag)
  );
}
