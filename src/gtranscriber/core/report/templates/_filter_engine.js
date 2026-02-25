/* G-Transcriber Report: Filter Engine + Chart Builders */
(function () {
  "use strict";

  var DATA = { qa_pairs: [], transcriptions: [], runs: [] };
  var ALL_RUNS = [];
  var renderedTabs = {};
  var ACTIVE_THRESHOLDS = { validation: null, quality: null };
  var _thresholdCache = {};
  var _funnelCache = {};
  var activeRunId = null;
  var activeRunThreshold = 0.6;
  var qaDetailState = { page: 1, sortBy: "source_filename", sortOrder: "asc", totalPages: 1 };
  var qaRequestId = 0;
  var DEFAULT_SCORE_THRESHOLD = 0.6;
  var TRANS_TABLE_STATE = { page: 1, sortBy: "source_filename", sortOrder: "asc", totalPages: 1 };
  var BLOOM_COLORS = {
    remember: "#0173B2",
    understand: "#029E73",
    apply: "#D55E00",
    analyze: "#DE8F05",
    evaluate: "#CC3311",
    create: "#CC79A7",
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

  /* ===================== Auto-Pagination ===================== */

  /**
   * Fetch all pages from a paginated API endpoint and return a flat array.
   * Uses per_page=250 (backend max) and fetches remaining pages in parallel
   * (batched, max 4 concurrent).
   */
  async function fetchAllPages(baseUrl) {
    var separator = baseUrl.indexOf("?") === -1 ? "?" : "&";
    var firstUrl = baseUrl + separator + "per_page=250&page=1";
    var first = await fetch(firstUrl).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
    var items = first.items || [];
    var totalPages = first.total_pages || 1;
    if (totalPages <= 1) return items;

    // Build remaining page URLs
    var urls = [];
    for (var p = 2; p <= totalPages; p++) {
      urls.push(baseUrl + separator + "per_page=250&page=" + p);
    }

    // Fetch in batches of 4
    var BATCH = 4;
    for (var i = 0; i < urls.length; i += BATCH) {
      var batch = urls.slice(i, i + BATCH);
      var pages = await Promise.all(
        batch.map(function (url) {
          return fetch(url).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          });
        })
      );
      pages.forEach(function (page) {
        items = items.concat(page.items || []);
      });
    }
    return items;
  }

  /* ===================== Init ===================== */

  function init() {
    initFilters();
    initTabs();
    initSubTabs();
    initExportHtmlButton();
    loadDashboardData();
  }

  function initExportHtmlButton() {
    var btn = document.getElementById("btn-export-html");
    if (!btn) return;
    if (window.__EMBEDDED_DATA__) {
      btn.style.display = "none";
      return;
    }
    btn.onclick = function () {
      var sel = document.getElementById("run-selector");
      var activeRun = sel ? sel.value : null;
      if (activeRun) {
        window.location.href = "/api/export/html/" + encodeURIComponent(activeRun);
      }
    };
  }

  /* ===================== API Data Loading ===================== */

  async function loadDashboardData() {
    if (window.__EMBEDDED_DATA__) {
      loadFromEmbeddedData(window.__EMBEDDED_DATA__);
      return;
    }
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

  function loadFromEmbeddedData(embedded) {
    ALL_RUNS = embedded.runs || [];
    DATA.qa_pairs = embedded.qa_pairs || [];
    DATA.transcriptions = embedded.transcriptions || [];
    DATA.runs = embedded.runs || [];
    populateRunSelector(ALL_RUNS);
    if (ALL_RUNS.length) {
      activeRunId = ALL_RUNS[0].pipeline_id;
      activeRunThreshold = DEFAULT_SCORE_THRESHOLD;
      var locs = unique(DATA.transcriptions.map(function (t) { return t.location || ""; }).filter(Boolean));
      var parts = unique(DATA.transcriptions.map(function (t) { return t.participant_name || ""; }).filter(Boolean));
      populateSelect("filter-location", locs);
      populateSelect("filter-participant", parts);
      updateSummaryCards(ALL_RUNS[0], DATA.qa_pairs, DATA.transcriptions);
      renderedTabs = {};
      ACTIVE_THRESHOLDS = { validation: DEFAULT_SCORE_THRESHOLD, quality: DEFAULT_SCORE_THRESHOLD };
      renderTab("overview", DATA);
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
    activeRunId = pipelineId;
    TRANS_TABLE_STATE = { page: 1, sortBy: "source_filename", sortOrder: "asc", totalPages: 1 };
    // Fetch and cache the run config threshold for score coloring
    try {
      var config = await fetch("/api/runs/" + encodeURIComponent(pipelineId) + "/config").then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      activeRunThreshold = (config.configs && config.configs.cep && config.configs.cep.validation_threshold != null)
        ? config.configs.cep.validation_threshold
        : DEFAULT_SCORE_THRESHOLD;
    } catch (e) {
      activeRunThreshold = DEFAULT_SCORE_THRESHOLD;
    }
    try {
      var results = await Promise.all([
        fetchAllPages("/api/qa?pipeline=" + encodeURIComponent(pipelineId)),
        fetchAllPages("/api/transcriptions?pipeline=" + encodeURIComponent(pipelineId)),
        fetch("/api/runs/" + encodeURIComponent(pipelineId)).then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        }),
      ]);
      var qaItems = results[0];
      var transItems = results[1];
      var runDetail = results[2];
      DATA.qa_pairs = qaItems;
      DATA.transcriptions = transItems;
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
        if (btn.dataset.subtab === "qa-detail" && activeRunId) {
          qaDetailState.page = 1;
          buildQADetailTable(activeRunId, getActiveFilters());
        } else if (btn.dataset.subtab === "qa-alerts" && activeRunId) {
          loadQAAlerts(activeRunId, getActiveFilters());
        } else if (btn.dataset.subtab === "trans-detail" && activeRunId) {
          loadTranscriptionDetailTable(activeRunId, TRANS_TABLE_STATE.page, TRANS_TABLE_STATE.sortBy, TRANS_TABLE_STATE.sortOrder);
        } else if (btn.dataset.subtab === "trans-alerts" && activeRunId) {
          loadTranscriptionAlerts(activeRunId);
        }
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
        if (activeRunId) {
          var qaActiveSubTab = document.querySelector("#tab-qa .sub-tab-nav button.active");
          if (qaActiveSubTab) {
            var activeFilters = getActiveFilters();
            if (qaActiveSubTab.dataset.subtab === "qa-detail") {
              qaDetailState.page = 1;
              buildQADetailTable(activeRunId, activeFilters);
            } else if (qaActiveSubTab.dataset.subtab === "qa-alerts") {
              loadQAAlerts(activeRunId, activeFilters);
            }
          }
        }
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
        // Also load active subtab if it's detail or alerts
        if (activeRunId) {
          var activeTransSubtab = document.querySelector("#tab-transcriptions .sub-tab-nav button.active");
          var transSubtabId = activeTransSubtab ? activeTransSubtab.dataset.subtab : "trans-charts";
          if (transSubtabId === "trans-detail") {
            loadTranscriptionDetailTable(activeRunId, TRANS_TABLE_STATE.page, TRANS_TABLE_STATE.sortBy, TRANS_TABLE_STATE.sortOrder);
          } else if (transSubtabId === "trans-alerts") {
            loadTranscriptionAlerts(activeRunId);
          }
        }
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
      case "config": {
        var sel = document.getElementById("run-selector");
        loadConfigTab(sel ? sel.value : "");
        break;
      }
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

  /* ===================== Export Buttons ===================== */

  function buildExportButton(dataType, pipelineId) {
    if (window.__EMBEDDED_DATA__) return "";
    var params = new URLSearchParams({ type: dataType });
    if (pipelineId) params.set("pipeline", pipelineId);
    var filters = getActiveFilters();
    if (filters.validity === "valid") params.set("is_valid", "true");
    else if (filters.validity === "invalid") params.set("is_valid", "false");
    if (filters.minScore > 0) params.set("min_score", filters.minScore);
    var url = "/api/export/csv?" + params.toString();
    // Use data attribute to avoid inline onclick XSS risk; listener attached by delegateExportClicks
    return '<button class="export-btn" data-export-url="' + esc(url) + '">&#8595; Download CSV</button>';
  }

  function delegateExportClicks(container) {
    if (!container) return;
    container.querySelectorAll(".export-btn[data-export-url]").forEach(function (btn) {
      if (btn._exportBound) return;
      btn._exportBound = true;
      btn.addEventListener("click", function () {
        window.location.href = btn.dataset.exportUrl;
      });
    });
  }

  function buildRunSummaryTable(runs, divId) {
    var el = document.getElementById(divId);
    if (!el) return;
    var html = buildExportButton("runs", activeRunId);
    html +=
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
    delegateExportClicks(el);
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
      addThresholdOverlay(layout, threshold);
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
      // Scope to the first subplot (Overall Score) only, matching Python's row=1, col=1
      addThresholdOverlay(layout, threshold, "y", "x domain");
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

  function addThresholdOverlay(layout, threshold, yref, xref) {
    yref = yref || "y";
    xref = xref || "paper";
    layout.shapes = (layout.shapes || []).concat([{
      type: "line",
      y0: threshold, y1: threshold,
      x0: 0, x1: 1,
      xref: xref, yref: yref,
      line: { color: "#CC3311", width: 2, dash: "dash" },
    }]);
    layout.annotations = (layout.annotations || []).concat([{
      x: 1, xref: xref,
      y: threshold, yref: yref,
      text: "Threshold: " + threshold.toFixed(2),
      showarrow: false,
      font: { color: "#CC3311", size: 11 },
      xanchor: "right",
    }]);
  }

  async function getRunThresholds(pipelineId) {
    if (_thresholdCache[pipelineId]) return _thresholdCache[pipelineId];
    try {
      var config = await fetch("/api/runs/" + encodeURIComponent(pipelineId) + "/config").then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      var result = {
        validation: (config.configs && config.configs.cep && config.configs.cep.validation_threshold != null)
          ? config.configs.cep.validation_threshold : null,
        quality: (config.configs && config.configs.transcription && config.configs.transcription.quality_threshold != null)
          ? config.configs.transcription.quality_threshold : null,
      };
      _thresholdCache[pipelineId] = result;
      return result;
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
    var pid = run.pipeline_id;
    if (_funnelCache[pid]) {
      _renderFunnelFromData(_funnelCache[pid], divId);
      return;
    }
    fetch("/api/funnel/" + encodeURIComponent(pid))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (funnel) {
        _funnelCache[pid] = funnel;
        _renderFunnelFromData(funnel, divId);
      })
      .catch(function (e) {
        console.warn("Could not build funnel chart:", e);
        plotReact(divId, [], { title: FUNNEL_TITLE + ": Unavailable", height: 400 });
      });
  }

  function _renderFunnelFromData(funnel, divId) {
    var FUNNEL_TITLE = "Data Processing Funnel";
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
  }

  /* ===================== Config Tab ===================== */

  async function loadConfigTab(pipelineId) {
    var content = document.getElementById("config-content");
    if (!content) return;
    if (!pipelineId) {
      content.innerHTML = '<p class="placeholder-text">Select a run to view its configuration.</p>';
      return;
    }
    content.innerHTML = '<p class="placeholder-text">Loading configuration...</p>';
    try {
      var config = await fetch(
        "/api/runs/" + encodeURIComponent(pipelineId) + "/config"
      ).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      renderConfigSections(config, content);
    } catch (e) {
      console.error("Failed to load config for", pipelineId, ":", e);
      content.innerHTML =
        '<p class="placeholder-text">Configuration data unavailable.</p>';
    }
  }

  function renderConfigSections(config, container) {
    var html = "";
    var stepLabels = {
      transcription: "Transcription",
      cep: "CEP / QA Generation",
    };
    Object.keys(config.configs || {}).forEach(function (step) {
      var label =
        stepLabels[step] ||
        step.charAt(0).toUpperCase() + step.slice(1).replace(/_/g, " ");
      var values = config.configs[step];
      var thresholds = (config.threshold_fields || {})[step] || [];
      html += renderConfigSection(label, values, thresholds);
    });
    if (config.hardware) {
      html += renderConfigSection("Hardware", config.hardware, []);
    }
    if (config.execution) {
      html += renderConfigSection("Execution Environment", config.execution, []);
    }
    if (!html) {
      html =
        '<p class="placeholder-text">No configuration data available for this run.</p>';
    }
    container.innerHTML = html;
  }

  function renderConfigSection(label, values, thresholds) {
    var html =
      '<details class="config-section" open>' +
      '<summary class="config-section-title">' +
      esc(label) +
      "</summary>" +
      '<div class="config-grid">';
    Object.keys(values).forEach(function (key) {
      var isThreshold = thresholds.indexOf(key) >= 0;
      var displayKey = key.replace(/_/g, " ");
      html +=
        '<div class="config-key">' +
        esc(displayKey) +
        (isThreshold ? ' <span class="threshold-badge">threshold</span>' : "") +
        "</div>" +
        '<div class="config-value' +
        (isThreshold ? " threshold" : "") +
        '">' +
        renderConfigValue(key, values[key]) +
        "</div>";
    });
    html += "</div></details>";
    return html;
  }

  function renderConfigValue(key, val) {
    if (val === null || val === undefined) {
      return '<span class="config-null">N/A</span>';
    }
    if (typeof val === "boolean") {
      return val
        ? '<span class="status-badge status-completed">Yes</span>'
        : '<span class="status-badge status-failed">No</span>';
    }
    if (key === "bloom_distribution" && typeof val === "object") {
      return renderBloomMiniBar(val);
    }
    if (typeof val === "object" && isNumericObject(val)) {
      return renderWeightsList(val);
    }
    if (typeof val === "object") {
      return "<code>" + esc(JSON.stringify(val)) + "</code>";
    }
    return esc(String(val));
  }

  function isNumericObject(obj) {
    var keys = Object.keys(obj);
    return keys.length > 0 && keys.every(function (k) { return typeof obj[k] === "number"; });
  }

  function renderWeightsList(obj) {
    var chips = Object.keys(obj).map(function (k) {
      return '<span>' + esc(k.replace(/_/g, " ") + ": " + obj[k].toFixed(2)) + "</span>";
    }).join("");
    return '<div class="weights-list">' + chips + "</div>";
  }

  function renderBloomMiniBar(dist) {
    var levels = ["remember", "understand", "analyze", "evaluate"];
    function getCount(l) {
      return dist[l] || dist[l.charAt(0).toUpperCase() + l.slice(1)] || 0;
    }
    var total = levels.reduce(function (s, l) { return s + getCount(l); }, 0);
    if (total === 0) return '<span class="config-null">N/A</span>';
    var bars = '<div class="bloom-mini-bars">';
    levels.forEach(function (l) {
      var count = getCount(l);
      var pct = (count / total) * 100;
      if (pct > 0) {
        bars +=
          '<div class="bar-' +
          l +
          '" style="width:' +
          pct +
          '%" title="' +
          esc(l + ": " + count) +
          '"></div>';
      }
    });
    bars += "</div>";
    return bars;
  }

  /* ===================== Transcription Detail Table ===================== */

  function getQualityThreshold(config) {
    return (config && config.configs && config.configs.transcription &&
      config.configs.transcription.quality_threshold != null)
      ? config.configs.transcription.quality_threshold : 0.5;
  }

  function scoreCell(value, threshold) {
    if (value === null || value === undefined) return "<td>N/A</td>";
    var cls = value >= 0.8 ? "score-high" : value >= threshold ? "score-medium" : "score-low";
    return '<td class="' + cls + '">' + value.toFixed(3) + "</td>";
  }

  function buildTranscriptionRowHtml(row, idx, threshold, prefix) {
    var validBadge = row.is_valid === true
      ? '<span class="status-badge status-completed">Yes</span>'
      : row.is_valid === false
      ? '<span class="status-badge status-failed">No</span>'
      : "N/A";
    return '<tr class="expandable" data-idx="' + idx + '" data-file="' + esc(row.source_filename) + '">' +
      "<td><strong>" + esc(row.source_filename) + "</strong></td>" +
      "<td>" + esc(row.participant_name || "") + "</td>" +
      "<td>" + esc(row.location || "") + "</td>" +
      "<td>" + esc(row.model_id || "") + "</td>" +
      "<td>" + esc(row.detected_language || "") + "</td>" +
      "<td>" + (row.processing_duration_sec != null ? row.processing_duration_sec.toFixed(1) : "N/A") + "</td>" +
      scoreCell(row.overall_quality, threshold) +
      scoreCell(row.script_match, threshold) +
      scoreCell(row.repetition, threshold) +
      scoreCell(row.segment_quality, threshold) +
      scoreCell(row.content_density, threshold) +
      "<td>" + validBadge + "</td>" +
      '<td><span class="issues-badge" id="' + prefix + '-issues-' + idx + '">?</span></td>' +
      "</tr>" +
      '<tr class="expanded-row" data-parent="' + idx + '" style="display:none">' +
      '<td colspan="13"><div id="' + prefix + '-detail-' + idx + '"><em>Loading detail…</em></div></td>' +
      "</tr>";
  }

  function buildPaginationHtml(currentPage, totalPages) {
    return '<div style="display:flex;align-items:center;gap:12px;margin:12px 0">' +
      '<button id="trans-prev-btn" ' + (currentPage <= 1 ? "disabled" : "") + '>« Prev</button>' +
      '<span>Page ' + currentPage + ' of ' + totalPages + '</span>' +
      '<button id="trans-next-btn" ' + (currentPage >= totalPages ? "disabled" : "") + '>Next »</button>' +
      "</div>";
  }

  async function loadTranscriptionDetailTable(pipelineId, page, sortBy, sortOrder) {
    var container = document.getElementById("trans-detail-container");
    if (!container) return;
    container.innerHTML = '<p class="placeholder-text">Loading…</p>';
    try {
      var filters = getActiveFilters();
      var url = "/api/transcriptions?pipeline=" + encodeURIComponent(pipelineId) +
        "&page=" + page + "&per_page=25" +
        "&sort_by=" + encodeURIComponent(sortBy) +
        "&sort_order=" + encodeURIComponent(sortOrder);
      if (filters.validity === "valid") url += "&is_valid=true";
      else if (filters.validity === "invalid") url += "&is_valid=false";
      if (filters.minScore > 0) url += "&min_score=" + filters.minScore;
      if (filters.search) url += "&search=" + encodeURIComponent(filters.search);

      var configUrl = "/api/runs/" + encodeURIComponent(pipelineId) + "/config";
      var results = await Promise.all([
        fetch(url).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }),
        fetch(configUrl).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
      ]);
      var data = results[0];
      var config = results[1];
      var threshold = getQualityThreshold(config);

      TRANS_TABLE_STATE.page = page;
      TRANS_TABLE_STATE.sortBy = sortBy;
      TRANS_TABLE_STATE.sortOrder = sortOrder;
      TRANS_TABLE_STATE.totalPages = data.pages || 1;

      buildTranscriptionDetailTable(data, threshold, container, pipelineId);
    } catch (e) {
      console.error("Failed to load transcription detail table:", e);
      container.innerHTML = '<p class="placeholder-text">Failed to load transcription data.</p>';
    }
  }

  function buildTranscriptionDetailTable(data, threshold, container, pipelineId) {
    var items = data.items || [];
    var totalPages = data.pages || 1;
    var currentPage = data.page || 1;

    var cols = [
      { label: "Source File", field: "source_filename", sortable: true },
      { label: "Participant", field: "participant_name", sortable: true },
      { label: "Location", field: "location", sortable: true },
      { label: "Model", field: "model_id", sortable: true },
      { label: "Language", field: "detected_language", sortable: true },
      { label: "Duration (s)", field: "processing_duration_sec", sortable: true },
      { label: "Overall", field: "overall_quality", sortable: true },
      { label: "Script Match", field: "script_match", sortable: true },
      { label: "Repetition", field: "repetition", sortable: true },
      { label: "Segment Qual.", field: "segment_quality", sortable: true },
      { label: "Content Dens.", field: "content_density", sortable: true },
      { label: "Valid", field: "is_valid", sortable: true },
      { label: "Issues", field: null, sortable: false },
    ];

    var s = TRANS_TABLE_STATE;
    var html = buildExportButton("transcriptions", pipelineId);
    html += '<table class="data-table"><thead><tr>';
    cols.forEach(function (col) {
      if (!col.sortable) {
        html += "<th>" + col.label + "</th>";
        return;
      }
      var indicator = col.field === s.sortBy ? (s.sortOrder === "asc" ? " ▲" : " ▼") : "";
      html += '<th data-field="' + col.field + '" style="cursor:pointer">' +
        col.label + '<span class="sort-indicator">' + indicator + "</span></th>";
    });
    html += "</tr></thead><tbody>";

    items.forEach(function (row, idx) {
      html += buildTranscriptionRowHtml(row, idx, threshold, "trans");
    });

    html += "</tbody></table>";

    // Pagination controls
    html += buildPaginationHtml(currentPage, totalPages);

    container.innerHTML = html;
    delegateExportClicks(container);

    // Sort header click handlers
    container.querySelectorAll("th[data-field]").forEach(function (th) {
      th.onclick = function () {
        var field = th.dataset.field;
        var newOrder = (s.sortBy === field && s.sortOrder === "asc") ? "desc" : "asc";
        loadTranscriptionDetailTable(pipelineId, 1, field, newOrder);
      };
    });

    // Pagination button handlers
    var prevBtn = document.getElementById("trans-prev-btn");
    var nextBtn = document.getElementById("trans-next-btn");
    if (prevBtn) {
      prevBtn.onclick = function () {
        if (currentPage > 1) loadTranscriptionDetailTable(pipelineId, currentPage - 1, s.sortBy, s.sortOrder);
      };
    }
    if (nextBtn) {
      nextBtn.onclick = function () {
        if (currentPage < totalPages) loadTranscriptionDetailTable(pipelineId, currentPage + 1, s.sortBy, s.sortOrder);
      };
    }

    // Expandable row handlers
    container.querySelectorAll("tr.expandable").forEach(function (row) {
      row.onclick = function () {
        var idx = row.dataset.idx;
        var filename = row.dataset.file;
        var detailRow = container.querySelector('tr[data-parent="' + idx + '"]');
        if (!detailRow) return;
        var isVisible = detailRow.style.display !== "none";
        if (isVisible) {
          detailRow.style.display = "none";
        } else {
          detailRow.style.display = "";
          expandTranscriptionRow(pipelineId, filename, idx, container);
        }
      };
    });
  }

  async function expandTranscriptionRow(pipelineId, sourceFilename, idx, container) {
    await expandTranscriptionDetailRow("trans", pipelineId, sourceFilename, idx);
  }

  async function expandTranscriptionDetailRow(prefix, pipelineId, sourceFilename, idx) {
    var detailDiv = document.getElementById(prefix + "-detail-" + idx);
    var badgeEl = document.getElementById(prefix + "-issues-" + idx);
    if (!detailDiv) return;
    try {
      var detail = await fetch(
        "/api/transcriptions/" + encodeURIComponent(pipelineId) + "/" + encodeURIComponent(sourceFilename)
      ).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });

      var issueCount = detail.issues_detected ? detail.issues_detected.length : 0;
      if (badgeEl) {
        badgeEl.textContent = issueCount;
        badgeEl.className = "issues-badge " + (issueCount > 0 ? "has-issues" : "no-issues");
      }

      var issuesHtml = issueCount > 0
        ? '<strong>Issues Detected:</strong><ul class="issues-list">' +
          detail.issues_detected.map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") +
          "</ul>"
        : '<strong>Issues Detected:</strong><p class="no-issues">No quality issues detected</p>';

      var rationaleHtml = detail.quality_rationale
        ? "<strong>Quality Rationale:</strong><p>" + esc(detail.quality_rationale) + "</p>"
        : "";

      var previewHtml = detail.transcription_text_preview
        ? "<strong>Transcription Preview:</strong><p>" + esc(detail.transcription_text_preview) + "</p>"
        : "";

      detailDiv.innerHTML =
        '<div style="padding:8px 0">' +
        issuesHtml +
        rationaleHtml +
        "<strong>Segments:</strong><p>" + detail.segment_count + " segments</p>" +
        previewHtml +
        "</div>";
    } catch (e) {
      console.error("Failed to load transcription detail for", sourceFilename, ":", e);
      if (detailDiv) detailDiv.innerHTML = "<em>Failed to load detail.</em>";
    }
  }

  /* ===================== Transcription Quality Alerts ===================== */

  var ALERTS_STATE = { allItems: [], shown: 0, threshold: 0.5 };

  async function loadTranscriptionAlerts(pipelineId) {
    var container = document.getElementById("trans-alerts-container");
    if (!container) return;
    container.innerHTML = '<p class="placeholder-text">Loading…</p>';
    try {
      var configUrl = "/api/runs/" + encodeURIComponent(pipelineId) + "/config";
      var config = await fetch(configUrl).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
      var threshold = getQualityThreshold(config);

      // Fetch both: below-threshold items AND explicitly invalid items (OR condition from spec).
      var baseUrl = "/api/transcriptions?pipeline=" + encodeURIComponent(pipelineId);
      var results = await Promise.all([
        fetchAllPages(baseUrl + "&max_score=" + threshold + "&sort_by=overall_quality&sort_order=asc"),
        fetchAllPages(baseUrl + "&is_valid=false").catch(function () { return []; }),
      ]);

      // Merge and deduplicate by source_filename
      var seen = {};
      var merged = [];
      results[0].concat(results[1]).forEach(function (item) {
        var key = item.pipeline_id + ":" + item.source_filename;
        if (!seen[key]) { seen[key] = true; merged.push(item); }
      });

      // Sort by overall_quality ascending (nulls last)
      merged.sort(function (a, b) {
        if (a.overall_quality == null) return 1;
        if (b.overall_quality == null) return -1;
        return a.overall_quality - b.overall_quality;
      });

      ALERTS_STATE.allItems = merged;
      ALERTS_STATE.shown = 0;
      ALERTS_STATE.threshold = threshold;

      renderTranscriptionAlertsTable(container, pipelineId, false);
    } catch (e) {
      console.error("Failed to load transcription alerts:", e);
      container.innerHTML = '<p class="placeholder-text">Failed to load quality alerts.</p>';
    }
  }

  function renderTranscriptionAlertsTable(container, pipelineId, append) {
    var allItems = ALERTS_STATE.allItems;
    var threshold = ALERTS_STATE.threshold;
    var PAGE_SIZE = 20;

    var header = '<p style="margin-bottom:12px"><strong>' + allItems.length +
      ' transcription' + (allItems.length !== 1 ? "s" : "") +
      '</strong> below quality threshold or invalid (<strong>' + threshold.toFixed(2) + "</strong>)</p>";

    if (!allItems.length) {
      container.innerHTML = header + buildExportButton("transcriptions", pipelineId) + '<p class="placeholder-text">No transcriptions below quality threshold.</p>';
      delegateExportClicks(container);
      return;
    }

    var startIdx = append ? ALERTS_STATE.shown : 0;
    var newItems = allItems.slice(startIdx, startIdx + PAGE_SIZE);
    ALERTS_STATE.shown = startIdx + newItems.length;

    var tableHeader = '<table class="data-table"><thead><tr>' +
      "<th>Source File</th><th>Participant</th><th>Location</th><th>Model</th><th>Language</th>" +
      "<th>Duration (s)</th><th>Overall</th><th>Script Match</th><th>Repetition</th>" +
      "<th>Segment Qual.</th><th>Content Dens.</th><th>Valid</th><th>Issues</th>" +
      "</tr></thead><tbody>";
    var tableFooter = "</tbody></table>";
    var showMoreHtml = ALERTS_STATE.shown < allItems.length
      ? '<button id="alerts-show-more-btn" style="margin-top:8px">Show More (' +
        (allItems.length - ALERTS_STATE.shown) + ' remaining)</button>'
      : "";

    if (append) {
      // Append rows to existing tbody and update Show More button
      var tbody = container.querySelector("table.data-table tbody");
      if (tbody) {
        var fragment = document.createElement("tbody");
        fragment.innerHTML = newItems.map(function (row, i) {
          return buildTranscriptionRowHtml(row, startIdx + i, threshold, "alerts");
        }).join("");
        while (fragment.firstChild) tbody.appendChild(fragment.firstChild);
      }
      var oldBtn = document.getElementById("alerts-show-more-btn");
      if (oldBtn) oldBtn.outerHTML = showMoreHtml;
    } else {
      var rowsHtml = newItems.map(function (row, i) {
        return buildTranscriptionRowHtml(row, startIdx + i, threshold, "alerts");
      }).join("");
      container.innerHTML = header + buildExportButton("transcriptions", pipelineId) + tableHeader + rowsHtml + tableFooter + showMoreHtml;
      delegateExportClicks(container);
    }

    // Show More button handler
    var showMoreBtn = document.getElementById("alerts-show-more-btn");
    if (showMoreBtn) {
      showMoreBtn.onclick = function () {
        renderTranscriptionAlertsTable(container, pipelineId, true);
      };
    }

    // Expandable row handlers (attach to all current expandable rows)
    container.querySelectorAll("tr.expandable").forEach(function (row) {
      if (row._expandBound) return; // avoid duplicate handlers
      row._expandBound = true;
      row.onclick = function () {
        var idx = row.dataset.idx;
        var filename = row.dataset.file;
        var detailRow = container.querySelector('tr[data-parent="' + idx + '"]');
        if (!detailRow) return;
        var isVisible = detailRow.style.display !== "none";
        if (isVisible) {
          detailRow.style.display = "none";
        } else {
          detailRow.style.display = "";
          expandAlertsRow(pipelineId, filename, idx);
        }
      };
    });
  }

  async function expandAlertsRow(pipelineId, sourceFilename, idx) {
    await expandTranscriptionDetailRow("alerts", pipelineId, sourceFilename, idx);
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
            fetchAllPages("/api/qa?pipeline=" + encodeURIComponent(runA)),
            fetchAllPages("/api/qa?pipeline=" + encodeURIComponent(runB)),
            fetchAllPages("/api/transcriptions?pipeline=" + encodeURIComponent(runA)),
            fetchAllPages("/api/transcriptions?pipeline=" + encodeURIComponent(runB)),
          ]);
          var qaAll = results[0].concat(results[1]);
          var transAll = results[2].concat(results[3]);
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

  /* ===================== QA Detail Table ===================== */

  async function buildQADetailTable(pipelineId, filters) {
    var container = document.getElementById("subtab-qa-detail");
    if (!container || !pipelineId) return;
    var currentRequest = ++qaRequestId;
    container.innerHTML = '<p class="placeholder-text">Loading QA pairs\u2026</p>';
    try {
      var params = new URLSearchParams({
        pipeline: pipelineId,
        page: qaDetailState.page,
        per_page: 25,
        sort_by: qaDetailState.sortBy,
        sort_order: qaDetailState.sortOrder,
      });
      if (filters.validity === "valid") params.set("is_valid", "true");
      if (filters.validity === "invalid") params.set("is_valid", "false");
      if (filters.minScore > 0) params.set("min_score", filters.minScore);
      if (filters.search) params.set("search", filters.search);
      var data = await fetch("/api/qa?" + params.toString()).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      if (currentRequest !== qaRequestId) return;
      var items = data.items || [];
      var totalPages = data.total_pages || 1;
      var total = data.total || 0;
      qaDetailState.totalPages = totalPages;
      container.innerHTML = renderQADetailTableHtml(items, total, totalPages);
      delegateExportClicks(container);
      attachQADetailHandlers(container, pipelineId);
    } catch (e) {
      if (currentRequest !== qaRequestId) return;
      console.error("Failed to load QA detail table:", e);
      container.innerHTML = '<p class="placeholder-text">Failed to load QA data. Please try again.</p>';
    }
  }

  function renderQADetailTableHtml(items, total, totalPages) {
    var cols = [
      { field: "source_filename", label: "Source File", sortable: true },
      { field: "bloom_level", label: "Bloom", sortable: true },
      { field: null, label: "Question", sortable: false },
      { field: null, label: "Answer", sortable: false },
      { field: "confidence", label: "Confidence", sortable: true },
      { field: "faithfulness", label: "Faithfulness", sortable: true },
      { field: "bloom_calibration", label: "Bloom Cal.", sortable: true },
      { field: "informativeness", label: "Informative.", sortable: true },
      { field: "self_containedness", label: "Self-Cont.", sortable: true },
      { field: "overall_score", label: "Overall", sortable: true },
      { field: "is_valid", label: "Valid", sortable: true },
    ];
    var html = buildExportButton("qa", activeRunId);
    html += '<table class="data-table"><thead><tr>';
    cols.forEach(function (col) {
      if (col.sortable) {
        var indicator = col.field === qaDetailState.sortBy
          ? (qaDetailState.sortOrder === "asc" ? " \u25b2" : " \u25bc")
          : " \u21c5";
        html += '<th data-field="' + col.field + '" class="sortable-header">'
          + esc(col.label)
          + '<span class="sort-indicator">' + indicator + "</span></th>";
      } else {
        html += "<th>" + esc(col.label) + "</th>";
      }
    });
    html += "</tr></thead><tbody>";
    var threshold = activeRunThreshold;
    items.forEach(function (item) {
      html += '<tr class="expandable" data-pipeline="' + esc(item.pipeline_id)
        + '" data-filename="' + esc(item.source_filename)
        + '" data-idx="' + (item.idx !== undefined ? item.idx : 0) + '">';
      html += "<td>" + esc(item.source_filename) + "</td>";
      html += "<td>" + bloomBadge(item.bloom_level) + "</td>";
      html += '<td><em class="expand-hint">Click row to expand</em></td>';
      html += '<td><em class="expand-hint">Click row to expand</em></td>';
      html += '<td class="' + scoreClass(item.confidence, threshold) + '">'
        + fmtScore(item.confidence, 2) + "</td>";
      html += '<td class="' + scoreClass(item.faithfulness, threshold) + '">'
        + fmtScore(item.faithfulness, 2) + "</td>";
      html += '<td class="' + scoreClass(item.bloom_calibration, threshold) + '">'
        + fmtScore(item.bloom_calibration, 2) + "</td>";
      html += '<td class="' + scoreClass(item.informativeness, threshold) + '">'
        + fmtScore(item.informativeness, 2) + "</td>";
      html += '<td class="' + scoreClass(item.self_containedness, threshold) + '">'
        + fmtScore(item.self_containedness, 2) + "</td>";
      var passBadge = item.is_valid
        ? '<span class="status-badge status-completed">Pass</span>'
        : '<span class="status-badge status-failed">Fail</span>';
      html += '<td class="' + scoreClass(item.overall_score, threshold) + '">'
        + fmtScore(item.overall_score, 3) + " " + passBadge + "</td>";
      html += "<td>" + (item.is_valid
        ? '<span class="status-badge status-completed">Yes</span>'
        : '<span class="status-badge status-failed">No</span>') + "</td>";
      html += "</tr>";
    });
    html += "</tbody></table>";
    html += buildPaginationHtml(qaDetailState.page, totalPages, total, "qa-detail-pagination");
    return html;
  }

  function attachQADetailHandlers(container, pipelineId) {
    container.querySelectorAll(".sortable-header").forEach(function (th) {
      th.onclick = function () {
        var field = th.dataset.field;
        if (qaDetailState.sortBy === field) {
          qaDetailState.sortOrder = qaDetailState.sortOrder === "asc" ? "desc" : "asc";
        } else {
          qaDetailState.sortBy = field;
          qaDetailState.sortOrder = "asc";
        }
        qaDetailState.page = 1;
        buildQADetailTable(pipelineId, getActiveFilters());
      };
    });
    var prevBtn = document.getElementById("qa-detail-pagination-prev");
    var nextBtn = document.getElementById("qa-detail-pagination-next");
    if (prevBtn) {
      prevBtn.onclick = function () {
        if (qaDetailState.page > 1) {
          qaDetailState.page--;
          buildQADetailTable(pipelineId, getActiveFilters());
        }
      };
    }
    if (nextBtn) {
      nextBtn.onclick = function () {
        if (qaDetailState.page < qaDetailState.totalPages) {
          qaDetailState.page++;
          buildQADetailTable(pipelineId, getActiveFilters());
        }
      };
    }
    container.querySelectorAll("tr.expandable").forEach(function (row) {
      row.onclick = function (e) {
        if (e.target.tagName === "BUTTON") return;
        expandQARow(row.dataset.pipeline, row.dataset.filename, parseInt(row.dataset.idx, 10), row);
      };
    });
  }

  async function expandQARow(pipelineId, sourceFilename, index, row) {
    var existing = row.nextElementSibling;
    if (existing && existing.classList.contains("qa-expanded-detail")) {
      existing.remove();
      row.classList.remove("row-expanded");
      return;
    }
    if (row.dataset.loading === "true") return;
    row.dataset.loading = "true";
    try {
      var detail = await fetch(
        "/api/qa/" + encodeURIComponent(pipelineId)
        + "/" + encodeURIComponent(sourceFilename)
        + "/" + index
      ).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      var colCount = row.cells.length;
      var html = '<tr class="qa-expanded-detail"><td colspan="' + colCount + '">'
        + '<div class="detail-panel">'
        + '<div class="detail-field"><strong>Question:</strong><p>' + esc(detail.question) + "</p></div>"
        + '<div class="detail-field"><strong>Answer:</strong><p>' + esc(detail.answer) + "</p></div>"
        + '<div class="detail-field"><strong>Context:</strong><p class="context-text">' + esc(detail.context) + "</p></div>";
      if (detail.reasoning_trace) {
        html += '<div class="detail-field"><strong>Reasoning Trace:</strong><p>' + esc(detail.reasoning_trace) + "</p></div>";
      }
      if (detail.tacit_inference) {
        html += '<div class="detail-field"><strong>Tacit Inference:</strong><p>' + esc(detail.tacit_inference) + "</p></div>";
      }
      if (detail.validation_rationale) {
        html += '<div class="detail-field"><strong>Judge Rationale:</strong><p>' + esc(detail.validation_rationale) + "</p></div>";
      }
      if (detail.generation_thinking) {
        html += '<div class="detail-field"><strong>Generation Thinking:</strong><p>' + esc(detail.generation_thinking) + "</p></div>";
      }
      html += "</div></td></tr>";
      row.insertAdjacentHTML("afterend", html);
      row.classList.add("row-expanded");
      delete row.dataset.loading;
    } catch (e) {
      console.error("Failed to load QA detail:", e);
      var colCount = row.cells.length;
      var errHtml = '<tr class="qa-expanded-detail"><td colspan="' + colCount + '">'
        + '<div class="detail-panel" style="border-left-color: #CC3311;">'
        + '<p style="color: #CC3311;">'
        + "\u26A0 Could not load detail \u2014 " + esc(e.message) + ". Click row to dismiss and retry."
        + "</p></div></td></tr>";
      row.insertAdjacentHTML("afterend", errHtml);
      row.classList.add("row-expanded");
      delete row.dataset.loading;
    }
  }

  function buildPaginationHtml(page, totalPages, total, id) {
    return '<div class="pagination" id="' + id + '">'
      + '<button id="' + id + '-prev"' + (page <= 1 ? " disabled" : "") + ">&lt; Prev</button>"
      + '<span class="page-info">Page ' + page + " of " + totalPages + " (" + total + " items)</span>"
      + '<button id="' + id + '-next"' + (page >= totalPages ? " disabled" : "") + ">Next &gt;</button>"
      + "</div>";
  }

  /* ===================== QA Alerts ===================== */

  async function loadQAAlerts(pipelineId, filters) {
    var container = document.getElementById("subtab-qa-alerts");
    if (!container || !pipelineId) return;
    container.innerHTML = '<p class="placeholder-text">Loading quality alerts\u2026</p>';
    var threshold = 0.6;
    try {
      var config = await fetch("/api/runs/" + encodeURIComponent(pipelineId) + "/config").then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      if (config.configs && config.configs.cep && config.configs.cep.validation_threshold != null) {
        threshold = config.configs.cep.validation_threshold;
      }
    } catch (e) {
      console.warn("Could not load run config for threshold:", e);
    }
    try {
      var params = new URLSearchParams({
        pipeline: pipelineId,
        max_score: threshold,
        sort_by: "overall_score",
        sort_order: "asc",
        per_page: 20,
        page: 1,
      });
      if (filters.validity === "valid") params.set("is_valid", "true");
      if (filters.validity === "invalid") params.set("is_valid", "false");
      if (filters.minScore > 0) params.set("min_score", filters.minScore);
      if (filters.search) params.set("search", filters.search);
      var data = await fetch("/api/qa?" + params.toString()).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      container.innerHTML = renderAlertsHtml(data.items || [], data.total || 0, data.total_pages || 1, threshold);
      delegateExportClicks(container);
      attachAlertsHandlers(container, pipelineId, threshold, filters);
    } catch (e) {
      console.error("Failed to load QA alerts:", e);
      container.innerHTML = '<p class="placeholder-text">Failed to load quality alerts.</p>';
    }
  }

  function renderAlertsHtml(items, total, totalPages, threshold) {
    var html = '<div class="alerts-header">'
      + '<span class="alerts-count">' + total + " items</span>"
      + "<span> below validation threshold (" + threshold.toFixed(2) + ")</span>"
      + "</div>";
    html += buildExportButton("qa", activeRunId);
    if (!items.length) {
      html += '<p class="placeholder-text">No QA pairs below threshold \u2014 all items pass quality checks.</p>';
      return html;
    }
    html += '<table class="data-table"><thead><tr>'
      + "<th>Source File</th><th>Bloom</th><th>Confidence</th>"
      + "<th>Faithfulness</th><th>Bloom Cal.</th><th>Informative.</th>"
      + "<th>Self-Cont.</th><th>Overall</th><th>Valid</th>"
      + "</tr></thead><tbody>";
    items.forEach(function (item) {
      html += buildAlertRowHtml(item, threshold);
    });
    html += "</tbody></table>";
    if (totalPages > 1) {
      html += '<div class="show-more-container">'
        + '<button id="qa-alerts-show-more" data-page="2" data-threshold="' + threshold + '">Show More</button>'
        + "</div>";
    }
    return html;
  }

  function buildAlertRowHtml(item, threshold) {
    return '<tr class="expandable" data-pipeline="' + esc(item.pipeline_id)
      + '" data-filename="' + esc(item.source_filename)
      + '" data-idx="' + (item.idx !== undefined ? item.idx : 0) + '">'
      + "<td>" + esc(item.source_filename) + "</td>"
      + "<td>" + bloomBadge(item.bloom_level) + "</td>"
      + '<td class="' + scoreClass(item.confidence, threshold) + '">' + fmtScore(item.confidence, 2) + "</td>"
      + '<td class="' + scoreClass(item.faithfulness, threshold) + '">' + fmtScore(item.faithfulness, 2) + "</td>"
      + '<td class="' + scoreClass(item.bloom_calibration, threshold) + '">' + fmtScore(item.bloom_calibration, 2) + "</td>"
      + '<td class="' + scoreClass(item.informativeness, threshold) + '">' + fmtScore(item.informativeness, 2) + "</td>"
      + '<td class="' + scoreClass(item.self_containedness, threshold) + '">' + fmtScore(item.self_containedness, 2) + "</td>"
      + '<td class="score-low">' + fmtScore(item.overall_score, 3) + "</td>"
      + "<td>" + (item.is_valid
        ? '<span class="status-badge status-completed">Yes</span>'
        : '<span class="status-badge status-failed">No</span>') + "</td>"
      + "</tr>";
  }

  function attachAlertsHandlers(container, pipelineId, threshold, filters) {
    container.querySelectorAll("tr.expandable").forEach(function (row) {
      row.onclick = function (e) {
        if (e.target.tagName === "BUTTON") return;
        expandQARow(row.dataset.pipeline, row.dataset.filename, parseInt(row.dataset.idx, 10), row);
      };
    });
    var showMoreBtn = document.getElementById("qa-alerts-show-more");
    if (showMoreBtn) {
      showMoreBtn.onclick = async function () {
        var page = parseInt(showMoreBtn.dataset.page, 10);
        try {
          var params = new URLSearchParams({
            pipeline: pipelineId,
            max_score: threshold,
            sort_by: "overall_score",
            sort_order: "asc",
            per_page: 20,
            page: page,
          });
          if (filters.validity === "valid") params.set("is_valid", "true");
          if (filters.validity === "invalid") params.set("is_valid", "false");
          if (filters.search) params.set("search", filters.search);
          var data = await fetch("/api/qa?" + params.toString()).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          });
          var tbody = container.querySelector("tbody");
          if (tbody) {
            (data.items || []).forEach(function (item) {
              var template = document.createElement("template");
              template.innerHTML = buildAlertRowHtml(item, threshold);
              var tr = template.content.firstChild;
              tr.onclick = function (e) {
                if (e.target.tagName === "BUTTON") return;
                expandQARow(tr.dataset.pipeline, tr.dataset.filename, parseInt(tr.dataset.idx, 10), tr);
              };
              tbody.appendChild(tr);
            });
          }
          if (page >= data.total_pages) {
            showMoreBtn.parentElement.remove();
          } else {
            showMoreBtn.dataset.page = page + 1;
          }
        } catch (e) {
          console.error("Failed to load more alerts:", e);
        }
      };
    }
  }

  /* ===================== Helpers ===================== */

  function plotReact(divId, traces, layout) {
    var el = document.getElementById(divId);
    if (!el) return;
    Plotly.react(divId, traces, layout, { responsive: true });
  }

  function scoreClass(score, threshold) {
    if (score === null || score === undefined) return "";
    if (score >= 0.8) return "score-high";
    if (score >= (threshold !== undefined ? threshold : DEFAULT_SCORE_THRESHOLD)) return "score-medium";
    return "score-low";
  }

  function bloomBadge(level) {
    if (!level) return "";
    return '<span class="bloom-badge ' + esc(level) + '">'
      + esc(level.charAt(0).toUpperCase() + level.slice(1)) + "</span>";
  }

  function fmtScore(val, digits) {
    return val !== null && val !== undefined ? val.toFixed(digits) : "N/A";
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
