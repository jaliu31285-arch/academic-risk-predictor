(function () {
  var sel = document.getElementById("fi-model");
  var statusEl = document.getElementById("fi-status");
  var chartEl = document.getElementById("fi-chart");
  var featureZh = window.__FEATURE_ZH__ || {};
  if (!sel) return;

  function zhLabel(feature) {
    return featureZh[feature] || feature;
  }

  function renderChart(data) {
    if (!data || !data.success) {
      statusEl.textContent = (data && data.error) || "加载失败。";
      chartEl.style.display = "none";
      return;
    }
    var features = data.features || [];
    var importances = data.importances || [];
    // Prefer zh labels from the server, fall back to the client-side map.
    var zhList = (data.zh && data.zh.length === features.length)
      ? data.zh
      : features.map(function (f) { return zhLabel(f); });

    // Build rows and sort descending by importance
    var rows = [];
    for (var i = 0; i < features.length; i++) {
      rows.push({
        feature: features[i],
        label: zhList[i],
        importance: Number(importances[i]) || 0,
      });
    }
    rows.sort(function (a, b) { return b.importance - a.importance; });

    var maxVal = rows[0] ? rows[0].importance : 1;
    if (maxVal <= 0) maxVal = 1;

    // Build bar chart as a simple HTML table of rows (no external lib needed)
    chartEl.innerHTML = "";
    statusEl.textContent = "共 " + rows.length + " 个特征，按重要性降序排列。";

    var table = document.createElement("table");
    table.className = "fi-table";

    var thead = document.createElement("thead");
    thead.innerHTML =
      "<tr><th style='width:30%'>特征名</th>" +
      "<th style='width:10%'>重要性</th>" +
      "<th style='width:60%'>贡献度</th></tr>";
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var pct = (row.importance / maxVal) * 100;
      var impStr = row.importance.toFixed(4);
      tr.innerHTML =
        "<td class='fi-label'>" +
        "<span class='fi-name'>" + row.label + "</span>" +
        "<span class='fi-en'>(" + row.feature + ")</span>" +
        "</td>" +
        "<td class='fi-val'>" + impStr + "</td>" +
        "<td class='fi-bar-cell'>" +
        "<div class='fi-bar' style='width:" + pct.toFixed(1) + "%'></div>" +
        "</td>";
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    chartEl.appendChild(table);
    chartEl.style.display = "block";
  }

  function load() {
    if (!sel.value) {
      statusEl.textContent = "未选择模型。";
      chartEl.style.display = "none";
      return;
    }
    statusEl.textContent = "正在加载特征重要性...";
    chartEl.style.display = "none";
    fetch("/api/feature_importance?model=" + encodeURIComponent(sel.value))
      .then(function (res) { return res.json(); })
      .then(renderChart)
      .catch(function (err) { statusEl.textContent = String(err); });
  }

  sel.addEventListener("change", load);
  if ((window.__AVAILABLE__ || []).length) load();
})();
