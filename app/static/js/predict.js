(function () {
  // ------- state -------
  var features = (window.__FEATURES__ || []);
  var stats = (window.__STATS__ || {});
  var categorical = (window.__CATEGORICAL__ || {});
  var featureZh = (window.__FEATURE_ZH__ || {});
  var riskLabels = (window.__RISK_LABELS__ || {});
  var modelSelect = document.getElementById("model-select");
  var grid = document.getElementById("feature-grid");
  var resultBox = document.getElementById("result-box");
  var predictBtn = document.getElementById("btn-predict");
  var meanBtn = document.getElementById("btn-fill-mean");
  var resetBtn = document.getElementById("btn-reset");

  // ------- helpers -------
  function fmt(n) {
    if (n === null || n === undefined || isNaN(n)) return "-";
    return Number(n).toFixed(2);
  }

  function zhName(col) {
    return featureZh[col] || col;
  }

  function buildFeatureCard(col) {
    var isCat = Object.prototype.hasOwnProperty.call(categorical, col) &&
      Array.isArray(categorical[col]) && categorical[col].length > 0;
    var s = stats[col] || {};
    var card = document.createElement("div");
    card.className = "feature-card";
    card.dataset.feature = col;

    var head = document.createElement("div");
    head.className = "feature-card-head";

    var nameEl = document.createElement("div");
    nameEl.className = "feature-name";
    nameEl.textContent = zhName(col);

    var meta = document.createElement("div");
    meta.className = "feature-meta";
    if (isCat) {
      meta.innerHTML =
        '<span class="tag">分类</span>' +
        '<span>可选值: <b>' + categorical[col].join(" / ") + '</b></span>';
    } else {
      meta.innerHTML =
        '<span>最小 <b>' + fmt(s.min) + '</b></span>' +
        '<span>均值 <b>' + fmt(s.mean) + '</b></span>' +
        '<span>最大 <b>' + fmt(s.max) + '</b></span>';
    }

    head.appendChild(nameEl);
    head.appendChild(meta);

    var body = document.createElement("div");
    body.className = "feature-card-body";

    var input;
    if (isCat) {
      input = document.createElement("select");
      input.className = "cat-select";
      input.name = col;
      var emptyOpt = document.createElement("option");
      emptyOpt.value = "";
      emptyOpt.textContent = "选择一个类别（留空使用最常见值）";
      input.appendChild(emptyOpt);
      for (var k = 0; k < categorical[col].length; k++) {
        var opt = document.createElement("option");
        opt.value = categorical[col][k];
        opt.textContent = categorical[col][k];
        input.appendChild(opt);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.name = col;
      input.placeholder = "留空使用均值 (" + fmt(s.mean) + ")";
    }

    body.appendChild(input);
    card.appendChild(head);
    card.appendChild(body);
    return card;
  }

  function renderFeatures() {
    if (!features.length) {
      grid.innerHTML = '<div class="empty-state">尚未加载特征数据，或数据集中没有可用特征。</div>';
      return;
    }
    grid.innerHTML = "";
    for (var i = 0; i < features.length; i++) {
      grid.appendChild(buildFeatureCard(features[i]));
    }
  }

  function loadModels() {
    window.fetchJSON("/api/models").then(function (data) {
      if (!data.available || !data.available.length) {
        modelSelect.innerHTML = '<option value="">暂无可用模型</option>';
        return;
      }
      modelSelect.innerHTML = "";
      for (var i = 0; i < data.available.length; i++) {
        var name = data.available[i];
        var opt = document.createElement("option");
        opt.value = name;
        opt.textContent = (data.labels && data.labels[name]) || name;
        modelSelect.appendChild(opt);
      }
      if (data.current) modelSelect.value = data.current;
    }).catch(function () {
      modelSelect.innerHTML = '<option value="">加载模型失败</option>';
    });
  }

  function collectInputs() {
    var values = {};
    var inputs = grid.querySelectorAll('[name]');
    for (var i = 0; i < inputs.length; i++) {
      var inp = inputs[i];
      if (inp.value === "") continue;
      var name = inp.name;
      var isCat = Object.prototype.hasOwnProperty.call(categorical, name);
      if (isCat) {
        // For categorical columns, pass the raw label string so the
        // backend can map it to the correct numeric code.
        values[name] = inp.value;
      } else {
        values[name] = Number(inp.value);
      }
    }
    return values;
  }

  function setResultLoading() {
    resultBox.innerHTML =
      '<div class="result-loading">' +
        '<div class="loading-spinner"></div>' +
        '<div class="loading-text">正在预测，请稍候...</div>' +
      '</div>';
  }

  function setResultError(msg) {
    resultBox.innerHTML =
      '<div class="result-error">' +
        '<div class="error-title">预测失败</div>' +
        '<div class="error-body">' + (msg || "未知错误") + '</div>' +
      '</div>';
  }

  function maxIndex(arr) {
    var best = 0;
    for (var i = 1; i < arr.length; i++) {
      if (arr[i] > arr[best]) best = i;
    }
    return best;
  }

  function renderResult(data) {
    if (!data.success) {
      setResultError(data.error);
      return;
    }

    var riskLabel = data.risk_label || "未知";
    var level = data.risk_level;
    var levelClass = level === 0 ? "risk-low" : level === 1 ? "risk-mid" : "risk-high";

    var probs = data.probabilities || [];
    var labels = [riskLabels["0"], riskLabels["1"], riskLabels["2"]];
    if (!labels[0]) labels = ["低风险", "中风险", "高风险"];
    if (probs.length !== 3) {
      probs = [0, 0, 0];
      probs[level] = 1;
    }
    var maxIdx = maxIndex(probs);

    // 概率条
    var probsHtml = '<div class="prob-block">' +
      '<div class="prob-title">各风险等级概率</div>';
    for (var i = 0; i < 3; i++) {
      var pct = Math.round((probs[i] || 0) * 1000) / 10;
      var barClass = i === maxIdx ? "prob-bar top" : "prob-bar";
      var riskName = labels[i] || ("等级 " + i);
      probsHtml +=
        '<div class="prob-row">' +
          '<div class="prob-label">' + riskName + '</div>' +
          '<div class="prob-track"><div class="' + barClass + '" style="width:' + pct + '%"></div></div>' +
          '<div class="prob-value">' + pct.toFixed(1) + '%</div>' +
        '</div>';
    }
    probsHtml += '</div>';

    // 关键因素
    var factorsHtml = "";
    var factors = (data.explanation && data.explanation.key_factors) || [];
    if (factors.length) {
      factorsHtml = '<div class="factors-block">' +
        '<div class="factors-title">关键因素（与均值偏差较大）</div>';
      for (var j = 0; j < factors.length; j++) {
        var f = factors[j];
        var dir = f.direction === "高于均值" ? "dir-up" : "dir-down";
        var arrow = f.direction === "高于均值" ? "\u25B2" : "\u25BC";
        factorsHtml +=
          '<div class="factor-row">' +
            '<div class="factor-name">' + zhName(f.feature) + '</div>' +
            '<div class="factor-arrow ' + dir + '">' + arrow + ' ' + f.direction + '</div>' +
            '<div class="factor-values">输入值 <b>' + fmt(f.value) + '</b> · 均值 <b>' + fmt(f.mean) + '</b></div>' +
          '</div>';
      }
      factorsHtml += '</div>';
    } else if (data.explanation && data.explanation.note) {
      factorsHtml = '<div class="factors-block subtle">' +
        '<div class="factors-title">特征说明</div>' +
        '<div>' + data.explanation.note + '</div>' +
        '</div>';
    }

    var modelHtml = '<div class="model-used">使用模型：<b>' + (data.model_label || modelSelect.value || "-") + '</b></div>';

    // 分数预测卡片
    var scoreHtml = "";
    if (data.predicted_score !== undefined && data.predicted_score !== null && !isNaN(data.predicted_score)) {
      var rangeText = "";
      if ((data.score_min !== undefined && data.score_min !== null) &&
          (data.score_max !== undefined && data.score_max !== null)) {
        rangeText = '<div class="score-range">参考区间：<b>' + fmt(data.score_min) + ' ~ ' + fmt(data.score_max) + '</b></div>';
      }
      scoreHtml =
        '<div class="score-card">' +
          '<div class="score-title">预测学业分数</div>' +
          '<div class="score-value">' + fmt(data.predicted_score) + '</div>' +
          rangeText +
        '</div>';
    }

    resultBox.innerHTML =
      '<div class="risk-card ' + levelClass + '">' +
        '<div class="risk-title">预测风险等级</div>' +
        '<div class="risk-label">' + riskLabel + '</div>' +
      '</div>' +
      scoreHtml +
      probsHtml +
      factorsHtml +
      modelHtml;
  }

  // ------- events -------
  predictBtn.addEventListener("click", function () {
    if (!modelSelect.value) {
      setResultError("请先在上方选择一个预测模型。");
      return;
    }
    setResultLoading();
    var payload = {
      model: modelSelect.value,
      features: collectInputs(),
    };
    window.fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (data) {
      renderResult(data);
    }).catch(function (err) {
      setResultError(String(err && err.message || err));
    });
  });

  meanBtn.addEventListener("click", function () {
    var inputs = grid.querySelectorAll('[name]');
    for (var i = 0; i < inputs.length; i++) {
      var inp = inputs[i];
      var name = inp.name;
      var isCat = Object.prototype.hasOwnProperty.call(categorical, name);
      if (isCat) {
        if (inp.tagName === "SELECT" && inp.options.length > 1) {
          inp.value = inp.options[1].value;
        }
      } else {
        var s = stats[name];
        inp.value = s && typeof s.mean === "number" ? s.mean : 0;
      }
    }
  });

  resetBtn.addEventListener("click", function () {
    var inputs = grid.querySelectorAll('[name]');
    for (var i = 0; i < inputs.length; i++) inputs[i].value = "";
    resultBox.innerHTML =
      '<div class="result-placeholder">' +
        '<div class="placeholder-label">尚未预测</div>' +
        '<div class="placeholder-hint">在左侧输入特征并点击\"预测风险\"</div>' +
      '</div>';
  });

  // ------- init -------
  loadModels();
  renderFeatures();
})();
