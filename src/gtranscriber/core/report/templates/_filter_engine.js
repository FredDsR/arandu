/* G-Transcriber Report: Filter Engine + Chart Builders */
(function () {
  "use strict";

  var DATA = { qa_pairs: [], transcriptions: [], runs: [] };
  var ALL_RUNS = [];
  var renderedTabs = {};
  var ACTIVE_THRESHOLDS = { validation: null, quality: null };
  var BLOOM_COLORS = {
    remember: "#0173B2",
    understand: "#029E73",
    analyze: "#DE8F05",
    evaluate: "#CC3311",
  };
  var CRITERION_COLORS = {
    faithfulness: "#0173B2",
    bloom_calibration: "#DE8F05",
    informativeness: "#029E73",
    self_containedness: "#CC3311",
  };
  var COLORS = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC3311",
    "#6B1E7C",
    "#EE6677",
    "#6F4E37",
    "#949494",
  ];

  /* ===================== Init ===================== */

  function init() {
    initFilters();
    initTabs();
    initSubTabs();
    loadDashboardData();
  }

  /* ===================== API Data Loading ===================== */

  async function loadDashboardData() {
    try {
      var runs = await fetch("/api/runs").then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      ALL_RUNS = runs || [];
      populateRunSelector(runs);
      if (runs && runs.length) {
        await setActiveRun(runs[0].pipeline_id);
      }
    } catch (e) {
      // API not available; fall back to empty data
      console.error("Failed to load dashboard data:", e);
      applyAndUpdate();
    }
  }

  function populateRunSelector(runs) {
    var sel = document.getElementById("run-selector");
    if (!sel || !runs) return;
    sel.innerHTML = "";
    runs.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r.pipeline_id;
      opt.textContent = r.pipeline_id;
      sel.appendChild(opt);
    });
    sel.onchange = function () {
      setActiveRun(sel.value);
    };
    // Also populate pipeline filter
    populateSelect("filter-pipeline", runs.map(function (r) { return r.pipeline_id; }));
  }

  async function setActiveRun(pipelineId) {
    if (!pipelineId) return;
    try {
      var results = await Promise.all([
        fetch("/api/qa?pipeline=" + encodeURIComponent(pipelineId) + "&per_page=1000").then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        }),
        fetch("/api/transcriptions?pipeline=" + encodeURIComponent(pipelineId) + "&per_page=1000").then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        }),
        fetch("/api/runs/" + encodeURIComponent(pipelineId)).then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        }),
      ]);
      var qaData = results[0];
      var transData = results[1];
      var runDetail = results[2];
      DATA.qa_pairs = qaData.items || [];
      DATA.transcriptions = transData.items || [];
      DATA.runs = [runDetail];
      // Populate location/participant filters from fetched data
      var locs = unique(DATA.transcriptions.map(function (t) { return t.location || ""; }).filter(Boolean));
      var parts = unique(DATA.transcriptions.map(function (t) { return t.participant_name || ""; }).filter(Boolean));
      populateSelect("filter-location", locs);
      populateSelect("filter-participant", parts);
      updateSummaryCards(runDetail, DATA.qa_pairs, DATA.transcriptions);
      renderedTabs = {};
      var activeTab = document.querySelector("#tab-nav button.active");
      var tabId = activeTab ? activeTab.dataset.tab : "overview";
      renderedTabs[tabId] = true;
      // Fetch thresholds in parallel with rendering
      ACTIVE_THRESHOLDS = await getRunThresholds(pipelineId);
      renderTab(tabId, DATA);
    } catch (e) {
      console.error("Failed to load run data for", pipelineId, ":", e);
      applyAndUpdate();
    }
  }

  /* ===================== Filters ===================== */

  function initFilters() {
    // Bloom toggles
    var bloomDiv = document.getElementById("filter-bloom");
    if (bloomDiv) {
      ["remember", "understand", "analyze", "evaluate"].forEach(function (lvl) {
        var btn = document.createElement("button");
        btn.className = "bloom-toggle active";
        btn.dataset.level = lvl;
        btn.textContent = lvl.charAt(0).toUpperCase() + lvl.slice(1);
        btn.onclick = function () {
          btn.classList.toggle("active");
          applyAndUpdate();
        };
        bloomDiv.appendChild(btn);
      });
    }

    // Confidence slider
    var slider = document.getElementById("filter-confidence");
    if (slider) {
      slider.addEventListener("input", function () {
        var display = document.getElementById("confidence-value");
        if (display) display.textContent = parseFloat(slider.value).toFixed(2);
        applyAndUpdate();
      });
    }

    // Min validation score slider
    var minScore = document.getElementById("filter-min-score");
    if (minScore) {
      minScore.addEventListener("input", function () {
        var display = document.getElementById("min-score-value");
        if (display) display.textContent = parseFloat(minScore.value).toFixed(2);
        applyAndUpdate();
      });
    }

    // Validity dropdown
    var validity = document.getElementById("filter-validity");
    if (validity) validity.addEventListener("change", applyAndUpdate);

    // Search input
    var search = document.getElementById("filter-search");
    if (search) search.addEventListener("input", applyAndUpdate);

    // Select change handlers
    ["filter-pipeline", "filter-location", "filter-participant"].forEach(
      function (id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener("change", applyAndUpdate);
      }
    );

    // Reset button
    var resetBtn = document.getElementById("btn-reset-filters");
    if (resetBtn) {
      resetBtn.onclick = function () {
        ["filter-pipeline", "filter-location", "filter-participant"].forEach(
          function (id) {
            var el = document.getElementById(id);
            if (el) {
              for (var i = 0; i < el.options.length; i++)
                el.options[i].selected = false;
            }
          }
        );
        var s = document.getElementById("filter-confidence");
        if (s) {
          s.value = 0;
          var d = document.getElementById("confidence-value");
          if (d) d.textContent = "0.00";
        }
        var ms = document.getElementById("filter-min-score");
        if (ms) {
          ms.value = 0;
          var dm = document.getElementById("min-score-value");
          if (dm) dm.textContent = "0.00";
        }
        var v = document.getElementById("filter-validity");
        if (v) v.value = "";
        var sr = document.getElementById("filter-search");
        if (sr) sr.value = "";
        document.querySelectorAll(".bloom-toggle").forEach(function (b) {
          b.classList.add("active");
        });
        applyAndUpdate();
      };
    }
  }

  function populateSelect(id, values) {
    var el = document.getElementById(id);
    if (!el || !values) return;
    values.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      el.appendChild(opt);
    });
  }

  function getActiveFilters() {
    var filters = {};

    filters.pipelines = getSelectedValues("filter-pipeline");
    filters.locations = getSelectedValues("filter-location");
    filters.participants = getSelectedValues("filter-participant");

    var slider = document.getElementById("filter-confidence");
    filters.minConfidence = slider ? parseFloat(slider.value) : 0;

    var minScore = document.getElementById("filter-min-score");
    filters.minScore = minScore ? parseFloat(minScore.value) : 0;

    var validity = document.getElementById("filter-validity");
    filters.validity = validity ? validity.value : "";

    var search = document.getElementById("filter-search");
    filters.search = search ? search.value.trim().toLowerCase() : "";

    filters.bloomLevels = [];
    document.querySelectorAll(".bloom-toggle.active").forEach(function (btn) {
      filters.bloomLevels.push(btn.dataset.level);
    });

    return filters;
  }

  function getSelectedValues(id) {
    var el = document.getElementById(id);
    if (!el) return [];
    var vals = [];
    for (var i = 0; i < el.selectedOptions.length; i++) {
      vals.push(el.selectedOptions[i].value);
    }
    return vals;
  }

  function applyFilters(data, filters) {
    var qa = data.qa_pairs.filter(function (q) {
      if (
        filters.pipelines.length &&
        filters.pipelines.indexOf(q.pipeline_id) === -1
      )
        return false;
      if (
        filters.locations.length &&
        (!q.location || filters.locations.indexOf(q.location) === -1)
      )
        return false;
      if (
        filters.participants.length &&
        (!q.participant_name ||
          filters.participants.indexOf(q.participant_name) === -1)
      )
        return false;
      if (q.confidence !== null && q.confidence < filters.minConfidence)
        return false;
      if (
        filters.bloomLevels.length &&
        q.bloom_level &&
        filters.bloomLevels.indexOf(q.bloom_level) === -1
      )
        return false;
      if (filters.validity === "valid" && !q.is_valid) return false;
      if (filters.validity === "invalid" && q.is_valid) return false;
      if (filters.minScore > 0 && (q.overall_score === null || q.overall_score < filters.minScore))
        return false;
      if (filters.search) {
        var hay = ((q.source_filename || "") + " " + (q.participant_name || "")).toLowerCase();
        if (hay.indexOf(filters.search) === -1) return false;
      }
      return true;
    });

    var trans = data.transcriptions.filter(function (t) {
      if (
        filters.pipelines.length &&
        filters.pipelines.indexOf(t.pipeline_id) === -1
      )
        return false;
      if (
        filters.locations.length &&
        (!t.location || filters.locations.indexOf(t.location) === -1)
      )
        return false;
      if (
        filters.participants.length &&
        (!t.participant_name ||
          filters.participants.indexOf(t.participant_name) === -1)
      )
        return false;
      if (filters.validity === "valid" && !t.is_valid) return false;
      if (filters.validity === "invalid" && t.is_valid) return false;
      if (filters.search) {
        var hay = ((t.source_filename || "") + " " + (t.participant_name || "")).toLowerCase();
        if (hay.indexOf(filters.search) === -1) return false;
      }
      return true;
    });

    var runs = data.runs.filter(function (r) {
      if (
        filters.pipelines.length &&
        filters.pipelines.indexOf(r.pipeline_id) === -1
      )
        return false;
      return true;
    });

    return { qa_pairs: qa, transcriptions: trans, runs: runs };
  }

  /* ===================== Tabs ===================== */

  function initTabs() {
    document.querySelectorAll("#tab-nav button").forEach(function (btn) {
      btn.onclick = function () {
        switchTab(btn.dataset.tab);
      };
    });
  }

  function initSubTabs() {
    document.querySelectorAll(".sub-tab-nav button").forEach(function (btn) {
      btn.onclick = function () {
        var parent = btn.closest(".tab-panel");
        parent.querySelectorAll(".sub-tab-nav button").forEach(function (b) {
          b.classList.toggle("active", b === btn);
        });
        parent.querySelectorAll(".sub-tab-panel").forEach(function (p) {
          p.classList.toggle("active", p.id === "subtab-" + btn.dataset.subtab);
        });
      };
    });
  }

  function switchTab(tabId) {
    document
      .querySelectorAll("#tab-nav button")
      .forEach(function (b) {
        b.classList.toggle("active", b.dataset.tab === tabId);
      });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.classList.toggle("active", p.id === "tab-" + tabId);
    });

    if (!renderedTabs[tabId]) {
      renderedTabs[tabId] = true;
      var filters = getActiveFilters();
      var filtered = applyFilters(DATA, filters);
      renderTab(tabId, filtered);
    }
  }

  /* ===================== Update All ===================== */

  function applyAndUpdate() {
    var filters = getActiveFilters();
    var filtered = applyFilters(DATA, filters);
    var runSummary = filtered.runs.length ? filtered.runs[0] : null;
    updateSummaryCards(runSummary, filtered.qa_pairs, filtered.transcriptions);

    // Re-render all previously rendered tabs
    renderedTabs = {};
    var activeTab = document.querySelector("#tab-nav button.active");
    var tabId = activeTab ? activeTab.dataset.tab : "overview";
    renderedTabs[tabId] = true;
    renderTab(tabId, filtered);
  }

  function renderTab(tabId, filtered) {
    switch (tabId) {
      case "overview":
        buildRunSummaryTable(filtered.runs, "chart-run-summary-table");
        buildFunnelChart(filtered.runs, "chart-funnel");
        buildPipelineOverview(filtered.runs, "chart-pipeline-overview");
        buildRunTimeline(filtered.runs, "chart-run-timeline");
        break;
      case "qa":
        buildBloomDistribution(filtered.qa_pairs, "chart-bloom-distribution");
        buildValidationViolins(filtered.qa_pairs, "chart-validation-violins", ACTIVE_THRESHOLDS.validation);
        buildBloomValidationHeatmap(
          filtered.qa_pairs,
          "chart-bloom-validation-heatmap",
          ACTIVE_THRESHOLDS.validation
        );
        buildConfidenceDistribution(
          filtered.qa_pairs,
          "chart-confidence-distribution"
        );
        buildMultihopChart(filtered.qa_pairs, "chart-multihop");
        buildCorrelationHeatmap(
          filtered.qa_pairs,
          "chart-correlation-heatmap"
        );
        buildParallelCoordinates(
          filtered.qa_pairs,
          "chart-parallel-coordinates"
        );
        break;
      case "transcriptions":
        buildTranscriptionQuality(
          filtered.transcriptions,
          "chart-transcription-quality",
          ACTIVE_THRESHOLDS.quality
        );
        buildQualityRadar(
          filtered.qa_pairs,
          filtered.transcriptions,
          "chart-quality-radar"
        );
        break;
      case "source":
        buildLocationTreemap(
          filtered.transcriptions,
          "chart-location-treemap"
        );
        buildParticipantBreakdown(
          filtered.qa_pairs,
          filtered.transcriptions,
          "chart-participant-breakdown"
        );
        buildLocationQuality(filtered.qa_pairs, "chart-location-quality");
        buildDocumentTable(
          filtered.qa_pairs,
          filtered.transcriptions,
          "chart-document-table"
        );
        break;
      case "compare":
        initCompareTab();
        break;
    }
  }

  /* ===================== Summary Cards ===================== */

  function updateSummaryCards(runSummary, qaPairs, transcriptions) {
    var validTrans = transcriptions.filter(function (t) { return t.is_valid === true; }).length;
    setText("card-total-transcriptions", validTrans + "/" + transcriptions.length);

    var validQA = qaPairs.filter(function (q) { return q.is_valid; }).length;
    setText("card-total-qa", validQA + "/" + qaPairs.length);

    var scores = qaPairs
      .map(function (q) { return q.overall_score; })
      .filter(function (v) { return v !== null; });
    var avgScore = scores.length
      ? (scores.reduce(function (a, b) { return a + b; }, 0) / scores.length).toFixed(3)
      : "N/A";
    setText("card-avg-score", avgScore);

    var successRate = (runSummary && runSummary.success_rate != null)
      ? runSummary.success_rate.toFixed(1) + "%"
      : "N/A";
    setText("card-avg-success", successRate);

    var tqScores = transcriptions
      .map(function (t) { return t.overall_quality; })
      .filter(function (v) { return v !== null; });
    var avgTQ = tqScores.length
      ? (tqScores.reduce(function (a, b) { return a + b; }, 0) / tqScores.length).toFixed(3)
      : "N/A";
    setText("card-avg-trans-quality", avgTQ);

    // HOTS ratio (Analyze + Evaluate / total)
    var hots = qaPairs.filter(function (q) {
      return q.bloom_level === "analyze" || q.bloom_level === "evaluate";
    }).length;
    var hotsRatio = qaPairs.length
      ? ((hots / qaPairs.length) * 100).toFixed(1) + "%"
      : "N/A";
    setText("card-hots-ratio", hotsRatio);
  }

  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  /* ===================== Chart Builders ===================== */

  function buildRunSummaryTable(runs, divId) {
    var el = document.getElementById(divId);
    if (!el) return;
    var html =
      '<table class="data-table"><thead><tr>' +
      "<th>Pipeline ID</th><th>Steps</th><th>Status</th>" +
      "<th>Duration (s)</th><th>Success Rate</th><th>Items</th>" +
      "</tr></thead><tbody>";
    runs.forEach(function (r) {
      var statusClass = "status-" + (r.status || "unknown");
      html +=
        "<tr>" +
        "<td><strong>" + esc(r.pipeline_id) + "</strong></td>" +
        "<td>" + esc((r.steps_run || []).join(", ")) + "</td>" +
        '<td><span class="status-badge ' + statusClass + '">' + esc(r.status) + "</span></td>" +
        "<td>" + (r.duration_seconds != null ? r.duration_seconds.toFixed(1) : "N/A") + "</td>" +
        "<td>" + (r.success_rate != null ? r.success_rate.toFixed(1) + "%" : "N/A") + "</td>" +
        "<td>" + r.completed_items + "/" + r.total_items + "</td>" +
        "</tr>";
    });
    html += "</tbody></table>";
    el.innerHTML = html;
  }

  function buildPipelineOverview(runs, divId) {
    var filtered = runs.filter(function (r) {
      return r.success_rate !== null;
    });
    var ids = filtered.map(function (r) { return r.pipeline_id; });
    var rates = filtered.map(function (r) { return r.success_rate; });
    var durs = filtered.map(function (r) { return r.duration_seconds || 0; });

    var traces = [
      { x: ids, y: rates, type: "bar", name: "Success Rate (%)", marker: { color: COLORS[0] }, xaxis: "x", yaxis: "y" },
      { x: ids, y: durs, type: "bar", name: "Duration (s)", marker: { color: COLORS[1] }, xaxis: "x2", yaxis: "y2" },
    ];
    var layout = {
      grid: { rows: 1, columns: 2, pattern: "independent" },
      height: 400, showlegend: false, template: "plotly_white",
      title: "Pipeline Run Success Rate & Processing Duration",
      yaxis: { title: "Success Rate (%)", range: [0, 100] },
      yaxis2: { title: "Duration (s)" },
    };
    plotReact(divId, traces, layout);
  }

  function buildBloomDistribution(qa, divId) {
    var levels = ["remember", "understand", "analyze", "evaluate"];
    var byRun = {};
    qa.forEach(function (q) {
      if (!q.bloom_level || levels.indexOf(q.bloom_level) === -1) return;
      if (!byRun[q.pipeline_id]) byRun[q.pipeline_id] = { remember: 0, understand: 0, analyze: 0, evaluate: 0 };
      byRun[q.pipeline_id][q.bloom_level]++;
    });
    var pids = Object.keys(byRun).sort();
    var traces = levels.map(function (lvl) {
      return {
        name: lvl.charAt(0).toUpperCase() + lvl.slice(1),
        x: pids,
        y: pids.map(function (p) { return byRun[p][lvl]; }),
        type: "bar",
        marker: { color: BLOOM_COLORS[lvl] },
      };
    });
    plotReact(divId, traces, {
      barmode: "stack", title: "Bloom's Taxonomy Level Distribution per Pipeline Run",
      xaxis: { title: "Pipeline ID" }, yaxis: { title: "Number of QA Pairs" },
      height: 450, template: "plotly_white",
    });
  }

  function buildValidationViolins(qa, divId, threshold) {
    var criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var traces = [];
    criteria.forEach(function (c) {
      var scores = qa.filter(function (q) { return q[c] !== null; }).map(function (q) { return q[c]; });
      if (scores.length) {
        traces.push({
          type: "violin", y: scores,
          name: c.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }),
          marker: { color: CRITERION_COLORS[c] },
          box: { visible: true }, meanline: { visible: true },
        });
      }
    });
    var layout = {
      title: "LLM-as-a-Judge Validation Score Distributions",
      yaxis: { title: "Score (0-1)" }, height: 450, template: "plotly_white",
    };
    if (threshold != null) {
      layout.shapes = [{
        type: "line",
        y0: threshold, y1: threshold,
        x0: 0, x1: 1,
        xref: "paper",
        line: { color: "#CC3311", width: 2, dash: "dash" },
      }];
      layout.annotations = (layout.annotations || []).concat([{
        x: 1, xref: "paper",
        y: threshold,
        text: "Threshold: " + threshold.toFixed(2),
        showarrow: false,
        font: { color: "#CC3311", size: 11 },
        xanchor: "right",
      }]);
    }
    plotReact(divId, traces, layout);
  }

  function buildConfidenceDistribution(qa, divId) {
    var scores = qa.filter(function (q) { return q.confidence !== null; }).map(function (q) { return q.confidence; });
    plotReact(divId, [{
      x: scores, type: "histogram", nbinsx: 30,
      marker: { color: COLORS[0] }, name: "Confidence Score",
    }], {
      title: "Generation Confidence Score Distribution",
      xaxis: { title: "Confidence (0-1)" }, yaxis: { title: "Frequency" },
      height: 400, template: "plotly_white",
    });
  }

  function buildTranscriptionQuality(trans, divId, threshold) {
    var fields = [
      ["overall_quality", "Overall Score"],
      ["script_match", "Script Match"],
      ["repetition", "Repetition"],
      ["segment_quality", "Segment Quality"],
      ["content_density", "Content Density"],
    ];
    var traces = [];
    var annotations = [];
    fields.forEach(function (pair, i) {
      var field = pair[0], label = pair[1];
      var scores = trans.filter(function (t) { return t[field] !== null; }).map(function (t) { return t[field]; });
      var row = i < 3 ? 0 : 1, col = i < 3 ? i : i - 3;
      traces.push({
        x: scores, type: "histogram", nbinsx: 20,
        marker: { color: COLORS[i] }, showlegend: false,
        xaxis: "x" + (i > 0 ? i + 1 : ""), yaxis: "y" + (i > 0 ? i + 1 : ""),
      });
    });
    // Validity bar
    var valid = trans.filter(function (t) { return t.is_valid === true; }).length;
    var invalid = trans.filter(function (t) { return t.is_valid === false; }).length;
    traces.push({
      x: ["Valid", "Invalid"], y: [valid, invalid], type: "bar",
      marker: { color: [COLORS[2], COLORS[3]] }, showlegend: false,
      xaxis: "x6", yaxis: "y6",
    });
    var layout = {
      grid: { rows: 2, columns: 3, pattern: "independent" },
      height: 600, showlegend: false, template: "plotly_white",
      title: "ASR Transcription Quality Score Distributions",
    };
    if (threshold != null) {
      layout.shapes = [{
        type: "line",
        y0: threshold, y1: threshold,
        x0: 0, x1: 1,
        xref: "paper", yref: "y",
        line: { color: "#CC3311", width: 2, dash: "dash" },
      }];
      layout.annotations = (layout.annotations || []).concat([{
        x: 1, xref: "paper",
        y: threshold, yref: "y",
        text: "Threshold: " + threshold.toFixed(2),
        showarrow: false,
        font: { color: "#CC3311", size: 11 },
        xanchor: "right",
      }]);
    }
    plotReact(divId, traces, layout);
  }

  function buildMultihopChart(qa, divId) {
    var multi = qa.filter(function (q) { return q.is_multi_hop; }).length;
    var single = qa.length - multi;
    plotReact(divId, [{
      y: ["Multi-hop", "Single-hop"], x: [multi, single],
      type: "bar", orientation: "h",
      marker: { color: [COLORS[0], COLORS[1]] },
      text: [multi, single], textposition: "auto",
    }], {
      title: "Reasoning Complexity: Multi-hop vs Single-hop",
      xaxis: { title: "Number of QA Pairs" },
      height: 300, template: "plotly_white",
    });
  }

  function buildCorrelationHeatmap(qa, divId) {
    var fields = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness", "confidence"];
    var labels = fields.map(function (f) { return f.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }); });
    var rows = [];
    qa.forEach(function (q) {
      var vals = fields.map(function (f) { return q[f]; });
      if (vals.every(function (v) { return v !== null; })) rows.push(vals);
    });
    var n = fields.length;
    var corr = [];
    for (var i = 0; i < n; i++) {
      corr[i] = [];
      for (var j = 0; j < n; j++) {
        if (rows.length >= 3) {
          var xi = rows.map(function (r) { return r[i]; });
          var xj = rows.map(function (r) { return r[j]; });
          corr[i][j] = pearsonR(xi, xj);
        } else {
          corr[i][j] = i === j ? 1 : 0;
        }
      }
    }
    var text = corr.map(function (row) { return row.map(function (v) { return v.toFixed(2); }); });
    plotReact(divId, [{
      z: corr, x: labels, y: labels, text: text, texttemplate: "%{text}",
      type: "heatmap", colorscale: "RdBu", reversescale: true,
      zmin: -1, zmax: 1, colorbar: { title: "r" },
    }], {
      title: "Pearson Correlation: Validation Criteria & Confidence",
      height: 500, width: 600, template: "plotly_white",
    });
  }

  function buildQualityRadar(qa, trans, divId) {
    var criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var theta = criteria.map(function (c) { return c.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }); });
    theta.push("Transcription Quality");

    var pids = unique(qa.map(function (q) { return q.pipeline_id; }).concat(trans.map(function (t) { return t.pipeline_id; }))).sort();
    var traces = [];
    pids.forEach(function (pid, i) {
      var rqa = qa.filter(function (q) { return q.pipeline_id === pid; });
      var rtrans = trans.filter(function (t) { return t.pipeline_id === pid; });
      var vals = criteria.map(function (c) {
        var s = rqa.filter(function (q) { return q[c] !== null; }).map(function (q) { return q[c]; });
        return s.length ? mean(s) : 0;
      });
      var tq = rtrans.filter(function (t) { return t.overall_quality !== null; }).map(function (t) { return t.overall_quality; });
      vals.push(tq.length ? mean(tq) : 0);
      vals.push(vals[0]); // close polygon
      traces.push({
        type: "scatterpolar", r: vals, theta: theta.concat([theta[0]]),
        fill: "toself", name: pid,
        marker: { color: COLORS[i % COLORS.length] },
      });
    });
    plotReact(divId, traces, {
      polar: { radialaxis: { visible: true, range: [0, 1] } },
      title: "Mean Quality Profile per Pipeline Run",
      height: 500, template: "plotly_white",
    });
  }

  function buildParallelCoordinates(qa, divId) {
    var fields = ["confidence", "faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var bloomOrder = ["remember", "understand", "analyze", "evaluate"];
    var rows = [];
    qa.forEach(function (q) {
      var vals = {};
      var allPresent = true;
      fields.forEach(function (f) { if (q[f] === null) allPresent = false; else vals[f] = q[f]; });
      if (allPresent && bloomOrder.indexOf(q.bloom_level) >= 0) {
        vals._bi = bloomOrder.indexOf(q.bloom_level);
        rows.push(vals);
      }
    });
    if (!rows.length) { plotReact(divId, [], { title: "Multi-dimensional QA Quality Profile: No data", height: 400 }); return; }
    var dims = fields.map(function (f) {
      return {
        label: f.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }),
        values: rows.map(function (r) { return r[f]; }), range: [0, 1],
      };
    });
    plotReact(divId, [{
      type: "parcoords",
      line: {
        color: rows.map(function (r) { return r._bi; }),
        colorscale: bloomOrder.map(function (b, i) { return [i / 3, BLOOM_COLORS[b]]; }),
        showscale: true, cmin: 0, cmax: 3,
        colorbar: { title: "Bloom", tickvals: [0, 1, 2, 3], ticktext: bloomOrder.map(function (b) { return b.charAt(0).toUpperCase() + b.slice(1); }) },
      },
      dimensions: dims,
    }], { title: "Multi-dimensional QA Quality Profile (colored by Bloom level)", height: 500, template: "plotly_white" });
  }

  function buildRunTimeline(runs, divId) {
    var dated = runs.filter(function (r) { return r.created_at && r.success_rate !== null; })
      .sort(function (a, b) { return (a.created_at || "").localeCompare(b.created_at || ""); });
    plotReact(divId, [{
      x: dated.map(function (r) { return r.created_at; }),
      y: dated.map(function (r) { return r.success_rate; }),
      mode: "lines+markers", type: "scatter",
      marker: { size: dated.map(function (r) { return Math.max(8, Math.min(30, (r.total_items || 1) * 2)); }), color: COLORS[0] },
      text: dated.map(function (r) { return r.pipeline_id; }),
      hovertemplate: "<b>%{text}</b><br>Success Rate: %{y:.1f}%<br>Date: %{x}<extra></extra>",
    }], {
      title: "Pipeline Run Timeline (marker size = item count)",
      xaxis: { title: "Date" }, yaxis: { title: "Success Rate (%)", range: [0, 100] },
      height: 400, template: "plotly_white",
    });
  }

  function buildParticipantBreakdown(qa, trans, divId) {
    var docCounts = {}, qaCounts = {};
    trans.forEach(function (t) { var n = t.participant_name || "Unknown"; docCounts[n] = (docCounts[n] || 0) + 1; });
    qa.forEach(function (q) { var n = q.participant_name || "Unknown"; qaCounts[n] = (qaCounts[n] || 0) + 1; });
    var parts = unique(Object.keys(docCounts).concat(Object.keys(qaCounts))).sort();
    plotReact(divId, [
      { name: "Documents", x: parts, y: parts.map(function (p) { return docCounts[p] || 0; }), type: "bar", marker: { color: COLORS[0] } },
      { name: "QA Pairs", x: parts, y: parts.map(function (p) { return qaCounts[p] || 0; }), type: "bar", marker: { color: COLORS[1] } },
    ], {
      barmode: "group", title: "Documents & QA Pairs per Participant",
      xaxis: { title: "Participant" }, yaxis: { title: "Count" },
      height: 450, template: "plotly_white",
    });
  }

  function buildLocationTreemap(trans, divId) {
    var labels = ["All"], parents = [""], values = [0];
    var locPart = {};
    trans.forEach(function (t) {
      var loc = t.location || "Unknown", part = t.participant_name || "Unknown";
      if (!locPart[loc]) locPart[loc] = {};
      locPart[loc][part] = (locPart[loc][part] || 0) + 1;
    });
    Object.keys(locPart).sort().forEach(function (loc) {
      labels.push(loc); parents.push("All"); values.push(0);
      Object.keys(locPart[loc]).sort().forEach(function (part) {
        labels.push(part + " (" + loc + ")");
        parents.push(loc);
        values.push(locPart[loc][part]);
      });
    });
    plotReact(divId, [{
      type: "treemap", labels: labels, parents: parents, values: values,
      branchvalues: "total", marker: { colorscale: "Blues" },
    }], { title: "Source Hierarchy: Location > Participant > Document Count", height: 500, template: "plotly_white" });
  }

  function buildBloomValidationHeatmap(qa, divId, threshold) {
    var levels = ["remember", "understand", "analyze", "evaluate"];
    var criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var cLabels = criteria.map(function (c) { return c.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }); });
    var data = {};
    levels.forEach(function (b) { data[b] = {}; criteria.forEach(function (c) { data[b][c] = []; }); });
    qa.forEach(function (q) {
      if (levels.indexOf(q.bloom_level) >= 0) {
        criteria.forEach(function (c) { if (q[c] !== null) data[q.bloom_level][c].push(q[c]); });
      }
    });
    var z = [], text = [];
    levels.forEach(function (b) {
      var zRow = [], tRow = [];
      criteria.forEach(function (c) {
        var s = data[b][c];
        if (s.length) {
          var m = mean(s), sd = stddev(s);
          zRow.push(m);
          var cellText = m.toFixed(2) + "\n+/-" + sd.toFixed(2);
          if (threshold != null && m < threshold) {
            cellText = "⚠ " + cellText;
          }
          tRow.push(cellText);
        } else { zRow.push(0); tRow.push("N/A"); }
      });
      z.push(zRow); text.push(tRow);
    });
    plotReact(divId, [{
      z: z, x: cLabels, y: levels.map(function (l) { return l.charAt(0).toUpperCase() + l.slice(1); }),
      text: text, texttemplate: "%{text}", type: "heatmap",
      colorscale: "YlOrRd", zmin: 0, zmax: 1, colorbar: { title: "Mean Score" },
    }], { title: "Mean Validation Score by Bloom Level and Criterion", height: 400, template: "plotly_white" });
  }

  function buildLocationQuality(qa, divId) {
    var byLoc = {};
    qa.forEach(function (q) {
      if (q.overall_score !== null) {
        var loc = q.location || "Unknown";
        if (!byLoc[loc]) byLoc[loc] = [];
        byLoc[loc].push(q.overall_score);
      }
    });
    var traces = [];
    Object.keys(byLoc).sort().forEach(function (loc, i) {
      traces.push({
        type: "violin", y: byLoc[loc], name: loc,
        marker: { color: COLORS[i % COLORS.length] },
        box: { visible: true }, meanline: { visible: true },
      });
    });
    plotReact(divId, traces, {
      title: "Overall Validation Score Distribution by Recording Location",
      yaxis: { title: "Overall Score (0-1)" },
      height: 450, template: "plotly_white",
    });
  }

  function buildDocumentTable(qa, trans, divId) {
    var el = document.getElementById(divId);
    if (!el) return;

    // Group by source_filename
    var docs = {};
    trans.forEach(function (t) {
      var key = t.pipeline_id + ":" + t.source_filename;
      if (!docs[key]) docs[key] = { filename: t.source_filename, pipeline_id: t.pipeline_id, participant: t.participant_name || "", location: t.location || "", date: t.recording_date || "", is_valid: t.is_valid, qa_pairs: [] };
    });
    qa.forEach(function (q) {
      var key = q.pipeline_id + ":" + q.source_filename;
      if (!docs[key]) docs[key] = { filename: q.source_filename, pipeline_id: q.pipeline_id, participant: q.participant_name || "", location: q.location || "", date: q.recording_date || "", is_valid: null, qa_pairs: [] };
      docs[key].qa_pairs.push(q);
    });

    var rows = Object.keys(docs).sort().map(function (k) { return docs[k]; });

    var html = '<table class="data-table"><thead><tr>' +
      "<th>Filename</th><th>Pipeline</th><th>Participant</th><th>Location</th>" +
      "<th>Date</th><th>QA Pairs</th><th>Mean Score</th><th>Bloom Dist.</th><th>Valid</th>" +
      "</tr></thead><tbody>";

    rows.forEach(function (doc, idx) {
      var scores = doc.qa_pairs.filter(function (q) { return q.overall_score !== null; }).map(function (q) { return q.overall_score; });
      var meanScore = scores.length ? mean(scores).toFixed(3) : "N/A";
      var bloomDist = { remember: 0, understand: 0, analyze: 0, evaluate: 0 };
      doc.qa_pairs.forEach(function (q) { if (q.bloom_level && bloomDist[q.bloom_level] !== undefined) bloomDist[q.bloom_level]++; });
      var total = doc.qa_pairs.length || 1;
      var miniBar = '<div class="bloom-mini-bars">' +
        Object.keys(bloomDist).map(function (b) {
          var pct = (bloomDist[b] / total * 100);
          return pct > 0 ? '<div class="bar-' + b + '" style="width:' + pct + '%"></div>' : "";
        }).join("") + "</div>";
      var validBadge = doc.is_valid === true ? '<span class="status-badge status-completed">Yes</span>' :
        doc.is_valid === false ? '<span class="status-badge status-failed">No</span>' : "N/A";

      html += '<tr class="expandable" data-idx="' + idx + '">' +
        "<td>" + esc(doc.filename) + "</td>" +
        "<td>" + esc(doc.pipeline_id) + "</td>" +
        "<td>" + esc(doc.participant) + "</td>" +
        "<td>" + esc(doc.location) + "</td>" +
        "<td>" + esc(doc.date) + "</td>" +
        "<td>" + doc.qa_pairs.length + "</td>" +
        "<td>" + meanScore + "</td>" +
        "<td>" + miniBar + "</td>" +
        "<td>" + validBadge + "</td>" +
        "</tr>";

      // Expandable QA pair rows (hidden by default)
      doc.qa_pairs.forEach(function (q) {
        html += '<tr class="expanded-row" data-parent="' + idx + '" style="display:none">' +
          '<td colspan="2" style="padding-left:40px">Q: ' + esc((q.bloom_level || "?").toUpperCase()) + "</td>" +
          "<td>" + (q.confidence !== null ? q.confidence.toFixed(2) : "N/A") + "</td>" +
          "<td>" + (q.faithfulness !== null ? q.faithfulness.toFixed(2) : "N/A") + "</td>" +
          "<td>" + (q.bloom_calibration !== null ? q.bloom_calibration.toFixed(2) : "N/A") + "</td>" +
          "<td>" + (q.informativeness !== null ? q.informativeness.toFixed(2) : "N/A") + "</td>" +
          "<td>" + (q.self_containedness !== null ? q.self_containedness.toFixed(2) : "N/A") + "</td>" +
          "<td>" + (q.overall_score !== null ? q.overall_score.toFixed(3) : "N/A") + "</td>" +
          '<td>' + (q.is_valid ? '<span class="status-badge status-completed">Yes</span>' : '<span class="status-badge status-failed">No</span>') + "</td>" +
          "</tr>";
      });
    });

    html += "</tbody></table>";
    el.innerHTML = html;

    // Toggle expandable rows
    el.querySelectorAll("tr.expandable").forEach(function (row) {
      row.onclick = function () {
        var idx = row.dataset.idx;
        var children = el.querySelectorAll('tr[data-parent="' + idx + '"]');
        var visible = children.length && children[0].style.display !== "none";
        children.forEach(function (c) { c.style.display = visible ? "none" : ""; });
      };
    });
  }

  async function getRunThresholds(pipelineId) {
    try {
      var config = await fetch("/api/runs/" + encodeURIComponent(pipelineId) + "/config").then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      return {
        validation: (config.configs && config.configs.cep && config.configs.cep.validation_threshold != null)
          ? config.configs.cep.validation_threshold : null,
        quality: (config.configs && config.configs.transcription && config.configs.transcription.quality_threshold != null)
          ? config.configs.transcription.quality_threshold : null,
      };
    } catch (e) {
      console.warn("Could not fetch thresholds for", pipelineId, ":", e);
      return { validation: null, quality: null };
    }
  }

  function buildFunnelChart(runs, divId) {
    var FUNNEL_TITLE = "Data Processing Funnel";
    var run = runs && runs.length ? runs[0] : null;
    if (!run) {
      plotReact(divId, [], { title: FUNNEL_TITLE + ": No data", height: 400 });
      return;
    }
    fetch("/api/funnel/" + encodeURIComponent(run.pipeline_id))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (funnel) {
        var stages = funnel.stages || [];
        if (!stages.length) {
          plotReact(divId, [], { title: FUNNEL_TITLE + ": No data", height: 400 });
          return;
        }
        var n = stages.length;
        var dropNodeIdx = n;
        var nodeLabels = stages.map(function (s) { return s.label + " (" + s.count + ")"; });
        nodeLabels.push("Failed/Invalid");
        var nodeColors = stages.map(function () { return "#029E73"; });
        nodeColors.push("#CC3311");
        var sources = [], targets = [], values = [], linkColors = [];
        for (var i = 0; i < n - 1; i++) {
          var next = stages[i + 1];
          if (next.count > 0) {
            sources.push(i); targets.push(i + 1); values.push(next.count);
            linkColors.push("rgba(2, 158, 115, 0.4)");
          }
          if (next.drop_count > 0) {
            sources.push(i); targets.push(dropNodeIdx); values.push(next.drop_count);
            linkColors.push("rgba(204, 51, 17, 0.4)");
          }
        }
        plotReact(divId, [{
          type: "sankey",
          orientation: "h",
          node: {
            label: nodeLabels,
            color: nodeColors,
            pad: 15,
            thickness: 20,
          },
          link: {
            source: sources,
            target: targets,
            value: values,
            color: linkColors,
          },
        }], {
          title: FUNNEL_TITLE,
          height: 400,
          template: "plotly_white",
        });
      })
      .catch(function (e) {
        console.warn("Could not build funnel chart:", e);
        plotReact(divId, [], { title: FUNNEL_TITLE + ": Unavailable", height: 400 });
      });
  }

  /* ===================== Compare Tab ===================== */

  function initCompareTab() {
    var selA = document.getElementById("compare-run-a");
    var selB = document.getElementById("compare-run-b");
    if (!selA || !selB) return;

    var pids = unique(ALL_RUNS.map(function (r) { return r.pipeline_id; })).sort();
    [selA, selB].forEach(function (sel) {
      sel.innerHTML = "";
      pids.forEach(function (pid) {
        var opt = document.createElement("option");
        opt.value = pid; opt.textContent = pid;
        sel.appendChild(opt);
      });
    });
    if (pids.length > 1) selB.value = pids[1];

    var btn = document.getElementById("btn-compare");
    if (btn) {
      btn.onclick = async function () {
        var runA = selA.value;
        var runB = selB.value;
        if (!runA || !runB) {
          alert("Please select both Run A and Run B before comparing.");
          return;
        }
        try {
          var results = await Promise.all([
            fetch("/api/qa?pipeline=" + encodeURIComponent(runA) + "&per_page=1000").then(function (r) {
              if (!r.ok) throw new Error("HTTP " + r.status);
              return r.json();
            }),
            fetch("/api/qa?pipeline=" + encodeURIComponent(runB) + "&per_page=1000").then(function (r) {
              if (!r.ok) throw new Error("HTTP " + r.status);
              return r.json();
            }),
            fetch("/api/transcriptions?pipeline=" + encodeURIComponent(runA) + "&per_page=1000").then(function (r) {
              if (!r.ok) throw new Error("HTTP " + r.status);
              return r.json();
            }),
            fetch("/api/transcriptions?pipeline=" + encodeURIComponent(runB) + "&per_page=1000").then(function (r) {
              if (!r.ok) throw new Error("HTTP " + r.status);
              return r.json();
            }),
          ]);
          var qaAll = (results[0].items || []).concat(results[1].items || []);
          var transAll = (results[2].items || []).concat(results[3].items || []);
          buildCrossRunComparison(qaAll, runA, runB, "chart-cross-run-comparison");
          buildCompareRadar(qaAll, transAll, runA, runB, "chart-compare-radar");
        } catch (e) {
          console.error("Failed to fetch comparison data:", e);
          buildCrossRunComparison(DATA.qa_pairs, runA, runB, "chart-cross-run-comparison");
          buildCompareRadar(DATA.qa_pairs, DATA.transcriptions, runA, runB, "chart-compare-radar");
        }
      };
      if (pids.length >= 2) {
        btn.click();
      }
    }
  }

  function buildCrossRunComparison(qa, runA, runB, divId) {
    var criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var pairsA = qa.filter(function (q) { return q.pipeline_id === runA; });
    var pairsB = qa.filter(function (q) { return q.pipeline_id === runB; });
    var traces = [];
    criteria.forEach(function (c, i) {
      var sA = pairsA.filter(function (q) { return q[c] !== null; }).map(function (q) { return q[c]; });
      var sB = pairsB.filter(function (q) { return q[c] !== null; }).map(function (q) { return q[c]; });
      var ax = i > 0 ? (i + 1) : "";
      if (sA.length) traces.push({ type: "violin", y: sA, name: runA, marker: { color: COLORS[0] }, legendgroup: runA, showlegend: i === 0, xaxis: "x" + ax, yaxis: "y" + ax, box: { visible: true }, meanline: { visible: true } });
      if (sB.length) traces.push({ type: "violin", y: sB, name: runB, marker: { color: COLORS[1] }, legendgroup: runB, showlegend: i === 0, xaxis: "x" + ax, yaxis: "y" + ax, box: { visible: true }, meanline: { visible: true } });
    });
    plotReact(divId, traces, {
      grid: { rows: 2, columns: 2, pattern: "independent" },
      title: "Validation Score Comparison: " + runA + " vs " + runB,
      height: 600, template: "plotly_white", violinmode: "overlay",
      annotations: criteria.map(function (c, i) {
        return { text: c.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }), xref: "paper", yref: "paper", x: (i % 2) * 0.5 + 0.25, y: 1.02 - Math.floor(i / 2) * 0.5, showarrow: false, font: { size: 13, color: "#333" } };
      }),
    });
  }

  function buildCompareRadar(qa, trans, runA, runB, divId) {
    var criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"];
    var theta = criteria.map(function (c) { return c.replace(/_/g, " ").replace(/\b\w/g, function (l) { return l.toUpperCase(); }); });
    theta.push("Transcription Quality");

    function getVals(pid) {
      var rqa = qa.filter(function (q) { return q.pipeline_id === pid; });
      var rtrans = trans.filter(function (t) { return t.pipeline_id === pid; });
      var vals = criteria.map(function (c) {
        var s = rqa.filter(function (q) { return q[c] !== null; }).map(function (q) { return q[c]; });
        return s.length ? mean(s) : 0;
      });
      var tq = rtrans.filter(function (t) { return t.overall_quality !== null; }).map(function (t) { return t.overall_quality; });
      vals.push(tq.length ? mean(tq) : 0);
      vals.push(vals[0]);
      return vals;
    }

    plotReact(divId, [
      { type: "scatterpolar", r: getVals(runA), theta: theta.concat([theta[0]]), fill: "toself", name: runA, marker: { color: COLORS[0] } },
      { type: "scatterpolar", r: getVals(runB), theta: theta.concat([theta[0]]), fill: "toself", name: runB, marker: { color: COLORS[1] } },
    ], {
      polar: { radialaxis: { visible: true, range: [0, 1] } },
      title: "Mean Quality Profile: " + runA + " vs " + runB,
      height: 500, template: "plotly_white",
    });
  }

  /* ===================== Helpers ===================== */

  function plotReact(divId, traces, layout) {
    var el = document.getElementById(divId);
    if (!el) return;
    Plotly.react(divId, traces, layout, { responsive: true });
  }

  function mean(arr) {
    return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length;
  }

  function stddev(arr) {
    var m = mean(arr);
    return Math.sqrt(arr.reduce(function (s, v) { return s + (v - m) * (v - m); }, 0) / arr.length);
  }

  function pearsonR(x, y) {
    var n = x.length;
    if (n < 2) return 0;
    var mx = mean(x), my = mean(y);
    var num = 0, dx = 0, dy = 0;
    for (var i = 0; i < n; i++) {
      num += (x[i] - mx) * (y[i] - my);
      dx += (x[i] - mx) * (x[i] - mx);
      dy += (y[i] - my) * (y[i] - my);
    }
    dx = Math.sqrt(dx); dy = Math.sqrt(dy);
    return dx === 0 || dy === 0 ? 0 : num / (dx * dy);
  }

  function unique(arr) {
    var seen = {};
    return arr.filter(function (v) { if (seen[v]) return false; seen[v] = true; return true; });
  }

  function esc(s) {
    if (!s) return "";
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  /* ===================== Boot ===================== */

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
