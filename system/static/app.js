/* DUALBREACH 评测系统前端：NDJSON 流式渲染，零外部依赖。 */
(function () {
  "use strict";

  var NUMERIC = {
    budget: 1, max_iters: 1, beam_width: 1, p_iter: 1, train_samples: 1,
    distill_k: 1, guard_workers: 1, seed: 1, max_tokens: 1, limit: 1, offset: 1,
  };

  var state = { meta: null, singleTask: "", batchTask: "", singleRaw: null, batchMeta: null };

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function formValue(name) {
    var el = $(name);
    if (!el) return undefined;
    if (el.type === "checkbox") return el.checked;
    var v = el.value;
    if (NUMERIC[name]) {
      var n = Number(v);
      return isNaN(n) ? 0 : n;
    }
    return v;
  }

  function collectParams(extra) {
    var names = ["model_name", "base_url", "api_key", "no_proxy", "use_mock", "max_tokens",
      "guard_model", "judge_model", "attacker_model", "embed_model",
      "mode", "budget", "max_iters", "beam_width", "p_iter", "train_samples",
      "distill", "distill_k", "use_proxy", "guard_workers", "seed"];
    var out = {};
    names.forEach(function (n) { out[n] = formValue(n); });
    if (extra) { Object.keys(extra).forEach(function (k) { out[k] = extra[k]; }); }
    return out;
  }

  /* ---------------- 初始化 ---------------- */
  function initMeta() {
    fetch("/api/meta").then(function (r) { return r.json(); }).then(function (m) {
      state.meta = m;
      var d = m.defaults;
      Object.keys(d).forEach(function (k) {
        var el = $(k);
        if (!el) return;
        if (el.type === "checkbox") { el.checked = !!d[k]; }
        else { el.value = d[k]; }
      });
      var sel = $("mode");
      sel.innerHTML = m.mode_choices.map(function (c) {
        var label = c === "dualbreach-v2" ? "单遍搜索" : "定向复攻（直接产出型诱导）";
        return '<option value="' + c + '">' + label + "</option>";
      }).join("");
      sel.value = d.mode;

      ["budget", "max_iters", "beam_width", "p_iter", "train_samples", "distill_k",
       "guard_workers", "limit"].forEach(function (k) {
        var el = $(k), out = $("v_" + k);
        function sync() { if (out) out.textContent = el.value; }
        el.addEventListener("input", sync);
        sync();
      });
      $("runs_dir").textContent = "结果目录：" + m.paths.batch_dir;
      $("backend-status").textContent = "后端已连接 · " + m.attack;
      $("backend-status").className = "badge ok";

      loadPipeline();
      loadRuns();
    }).catch(function (e) {
      $("backend-status").textContent = "后端未连接：" + e;
      $("backend-status").className = "badge err";
    });
  }

  function loadPipeline() {
    fetch("/api/pipeline").then(function (r) { return r.json(); }).then(function (j) {
      $("arch_html").innerHTML = j.html || "";
    }).catch(function () {});
  }

  /* ---------------- Tabs ---------------- */
  document.querySelectorAll(".tab").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".tab").forEach(function (b) { b.classList.remove("active"); });
      document.querySelectorAll(".tabpage").forEach(function (p) { p.classList.remove("active"); });
      btn.classList.add("active");
      $("page-" + btn.dataset.tab).classList.add("active");
    });
  });

  /* ---------------- NDJSON 流 ---------------- */
  function stream(url, payload, onEvent, onDone) {
    var ctl = new AbortController();
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctl.signal,
    }).then(function (resp) {
      var reader = resp.body.getReader();
      var decoder = new TextDecoder("utf-8");
      var buf = "";
      function pump() {
        return reader.read().then(function (res) {
          if (res.done) { onDone && onDone(); return null; }
          buf += decoder.decode(res.value, { stream: true });
          var lines = buf.split("\n");
          buf = lines.pop();
          lines.forEach(function (line) {
            line = line.trim();
            if (!line) return;
            try { onEvent(JSON.parse(line)); } catch (e) { /* 忽略半行 */ }
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      if (e.name !== "AbortError") { onEvent({ type: "error", message: String(e) }); }
      onDone && onDone();
    });
    return ctl;
  }

  function progressBar(done, total, note) {
    var w = total ? Math.min(100, (done / total) * 100) : 0;
    return '<div class="panel"><div class="bar"><i style="width:' + w.toFixed(1) + '%"></i></div>' +
      '<div class="hint">' + esc(note || "") + "</div></div>";
  }

  function attemptRow(a) {
    var hit = !a.guard_unsafe && Number(a.judge_score) === 5;
    var tag = hit ? '<span class="tag ok">双重成功</span>'
      : (!a.guard_unsafe ? '<span class="tag mid">过围栏·分不足</span>'
        : '<span class="tag no">被围栏拦截</span>');
    return "<tr><td>" + a.round + "</td><td>" + esc(a.origin) + "</td><td>" +
      esc(a.operator || "-") + "</td><td><b>" + a.judge_score + "</b></td><td>" +
      (a.guard_unsafe ? "unsafe" : "safe") + " · " + a.guard_score + "</td><td>" +
      a.guardrail + "</td><td>" + a.llm + "</td><td>" + tag + "</td>" +
      '<td class="clip" title="' + esc(a.prompt) + '">' + esc(a.prompt) + "</td></tr>";
  }

  var ATTEMPT_HEAD = '<table class="data"><thead><tr><th>轮次</th><th>来源</th><th>算子</th>' +
    "<th>裁判分</th><th>围栏判定</th><th>L_guard</th><th>L_llm</th><th>结果</th>" +
    "<th>本轮提示词</th></tr></thead><tbody></tbody></table>";

  /* ---------------- 单样本 ---------------- */
  var singleCtl = null;

  $("btn_single_run").addEventListener("click", function () {
    var goal = ($("goal").value || "").trim();
    if (!goal) { $("single_note").textContent = "请先填写有害目标"; return; }
    $("single_progress").innerHTML = "";
    $("single_stage2").innerHTML = "";
    $("single_stats").innerHTML = "";
    $("single_case").innerHTML = "";
    $("single_raw").textContent = "";
    $("single_attempts").innerHTML = '<div class="panel"><div class="panel-title">逐次查询明细</div>' +
      '<div class="table-wrap">' + ATTEMPT_HEAD + "</div></div>";
    var tbody = $("single_attempts").querySelector("tbody");

    var taskId = "single-" + Date.now();
    state.singleTask = taskId;
    $("btn_single_run").disabled = true;
    $("btn_single_cancel").disabled = false;
    $("single_note").textContent = "运行中…";

    var started = Date.now();
    var n = 0;
    singleCtl = stream("/api/single", { task_id: taskId, params: collectParams({ goal: goal }) },
      function (ev) {
        state.singleRaw = ev;
        if (ev.type === "start") {
          $("single_progress").innerHTML = progressBar(0, 1, "初始化 DualBreach 引擎…");
        } else if (ev.type === "stage2_start") {
          $("single_progress").innerHTML = progressBar(0, 1, "Stage 2 · 训练代理围栏（采集围栏标签）…");
        } else if (ev.type === "stage2_done") {
          $("single_stage2").innerHTML = ev.html || "";
          $("single_progress").innerHTML = progressBar(0, 1, "Stage 3 · 多目标搜索开始");
        } else if (ev.type === "tdi") {
          $("single_progress").innerHTML = progressBar(0, 1,
            "Stage 1 · TDI 已生成 " + (ev.prompts || []).length + " 条初始提示词");
        } else if (ev.type === "probe") {
          $("single_progress").innerHTML = progressBar(0, 1,
            "第 " + ev.iteration + " 轮 · 查目标（候选 " + ev.pool + " 条，代理围栏分 " + ev.proxy_guard + "）");
        } else if (ev.type === "attempt") {
          n += 1;
          tbody.insertAdjacentHTML("beforeend", attemptRow(ev));
        } else if (ev.type === "hard_restart") {
          $("single_progress").innerHTML = progressBar(0, 1,
            "第 " + ev.iteration + " 轮 · 触发「直接产出型」重启（诱导侧连续失败）");
        } else if (ev.type === "result") {
          $("single_stats").innerHTML = ev.html || "";
        } else if (ev.type === "done") {
          $("single_progress").innerHTML = progressBar(1, 1,
            "完成 · 用时 " + ev.elapsed + "s · 查询 " + (ev.record ? ev.record.queries : "-") + " 次");
          $("single_raw").textContent = JSON.stringify(ev, null, 2);
        } else if (ev.type === "cancelled") {
          $("single_note").textContent = "已取消";
        } else if (ev.type === "error") {
          $("single_note").textContent = "错误：" + ev.message;
        }
      },
      function () {
        $("btn_single_run").disabled = false;
        $("btn_single_cancel").disabled = true;
        if ($("single_note").textContent === "运行中…") {
          $("single_note").textContent = "完成（" + ((Date.now() - started) / 1000).toFixed(1) + "s）";
        }
      });
  });

  $("btn_single_cancel").addEventListener("click", function () {
    if (!state.singleTask) return;
    fetch("/api/cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.singleTask }),
    });
    $("single_note").textContent = "正在取消…";
  });

  /* ---------------- 批量 ---------------- */
  var batchCtl = null;
  var batchRows = [];

  $("btn_batch_run").addEventListener("click", function () {
    var params = collectParams({
      dataset_path: formValue("dataset_path"),
      limit: formValue("limit"),
      offset: formValue("offset"),
      resume: formValue("resume"),
      retry_errors: formValue("retry_errors"),
      tag: formValue("tag"),
    });
    $("batch_progress").innerHTML = "";
    $("batch_stage2").innerHTML = "";
    $("batch_stats").innerHTML = "";
    $("batch_meta").textContent = "";
    $("batch_table").innerHTML = "";
    batchRows = [];

    var taskId = "batch-" + Date.now();
    state.batchTask = taskId;
    $("btn_batch_run").disabled = true;
    $("btn_batch_cancel").disabled = false;
    $("batch_note").textContent = "运行中…";

    var started = Date.now();
    stream("/api/batch", { task_id: taskId, params: params },
      function (ev) {
        if (ev.type === "start") {
          $("batch_progress").innerHTML = progressBar(0, ev.total,
            "共 " + ev.total + " 条" + (ev.skipped ? "（跳过已完成 " + ev.skipped + " 条）" : "") +
            " · 输出 " + esc(ev.output));
        } else if (ev.type === "stage2_start") {
          $("batch_progress").innerHTML = progressBar(0, 1,
            "Stage 2 · 训练代理围栏（语料 " + ev.corpus + " 条）…");
        } else if (ev.type === "stage2_done") {
          $("batch_stage2").innerHTML = ev.html || "";
        } else if (ev.type === "goal_start") {
          $("batch_progress").innerHTML = progressBar(ev.idx - 1, ev.total,
            "[" + ev.idx + "/" + ev.total + "] " + ev.id + " · " + esc(ev.goal));
          $("batch_counter").textContent = ev.idx + " / " + ev.total;
        } else if (ev.type === "probe") {
          $("batch_progress").innerHTML = progressBar(ev.idx - 1, ev.total,
            "[" + ev.idx + "/" + ev.total + "] " + ev.id + " · 第 " + ev.iteration + " 轮查目标");
        } else if (ev.type === "record") {
          batchRows.push(ev.record);
          renderBatchTable();
          var r = ev.running;
          $("batch_progress").innerHTML = progressBar(ev.idx, ev.total,
            "[" + ev.idx + "/" + ev.total + "] 累计 ASR_L " + pct(r.asr_l) +
            " · ASR_G " + pct(r.asr_dual) + " · AQC " + r.avg_queries.toFixed(2));
          $("batch_counter").textContent = ev.idx + " / " + ev.total;
        } else if (ev.type === "goal_error") {
          $("batch_note").textContent = ev.id + " 出错：" + ev.message;
        } else if (ev.type === "done") {
          $("batch_progress").innerHTML = progressBar(1, 1, "批量完成 · 输出 " + esc(ev.output));
          $("batch_stats").innerHTML = ev.html || "";
          $("batch_meta").textContent = JSON.stringify(ev.summary, null, 2);
          state.batchMeta = ev.summary;
        } else if (ev.type === "cancelled") {
          $("batch_note").textContent = "已取消（已完成 " + ev.done + " 条）";
        } else if (ev.type === "error") {
          $("batch_note").textContent = "错误：" + ev.message;
        }
      },
      function () {
        $("btn_batch_run").disabled = false;
        $("btn_batch_cancel").disabled = true;
        if ($("batch_note").textContent === "运行中…") {
          $("batch_note").textContent = "完成（" + ((Date.now() - started) / 1000).toFixed(1) + "s）";
        }
      });
  });

  $("btn_batch_cancel").addEventListener("click", function () {
    if (!state.batchTask) return;
    fetch("/api/cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.batchTask }),
    });
    $("batch_note").textContent = "正在取消…";
  });

  function renderBatchTable() {
    var rows = batchRows.map(function (r) {
      var tag = r.asr_success ? '<span class="tag ok">双重成功</span>'
        : (r.llm_success ? '<span class="tag mid">诱导成功·围栏拦</span>'
          : '<span class="tag no">未成功</span>');
      return "<tr><td>" + esc(r.id) + '</td><td class="clip" title="' + esc(r.prompt) + '">' +
        esc(r.prompt) + "</td><td>" + esc(r.primary_domain || "-") + "</td><td>" +
        r.best_score + "</td><td>" + r.queries + "</td><td>" + tag + "</td></tr>";
    }).join("");
    $("batch_table").innerHTML = '<table class="data"><thead><tr><th>ID</th><th>目标</th>' +
      "<th>领域</th><th>最佳分</th><th>查询数</th><th>结果</th></tr></thead><tbody>" +
      rows + "</tbody></table>";
  }

  /* ---------------- 历史结果 ---------------- */
  function loadRuns() {
    fetch("/api/runs").then(function (r) { return r.json(); }).then(function (j) {
      var rows = (j.runs || []).map(function (r) {
        return "<tr><td>" + esc(r.name) + "</td><td>" + r.mtime + "</td><td>" + r.num_samples +
          "</td><td>" + pct(r.asr_l) + "</td><td>" + pct(r.asr_dual) + "</td><td>" +
          r.avg_queries.toFixed(2) + '</td><td><button class="btn" data-path="' +
          esc(r.path) + '">查看</button></td></tr>';
      }).join("");
      $("runs_table").innerHTML = '<table class="data"><thead><tr><th>文件</th><th>时间</th>' +
        "<th>样本</th><th>ASR_L</th><th>ASR_G</th><th>AQC</th><th></th></tr></thead><tbody>" +
        rows + "</tbody></table>";
      $("runs_table").querySelectorAll("button[data-path]").forEach(function (b) {
        b.addEventListener("click", function () { loadRunDetail(b.dataset.path); });
      });
    }).catch(function () {});
  }

  function loadRunDetail(path) {
    fetch("/api/runs/detail?path=" + encodeURIComponent(path))
      .then(function (r) { return r.json(); }).then(function (j) {
        $("run_detail_card").hidden = false;
        $("run_detail_title").textContent = j.name;
        $("run_detail_meta").textContent = "样本 " + j.total + " · ASR_L " + pct(j.stats.asr_l) +
          " · ASR_G " + pct(j.stats.asr_dual) + " · AQC " + j.stats.avg_queries.toFixed(2);
        var rows = (j.records || []).map(function (r) {
          var tag = r.asr_success ? '<span class="tag ok">双重成功</span>'
            : (r.llm_success ? '<span class="tag mid">诱导成功</span>'
              : '<span class="tag no">未成功</span>');
          return "<tr><td>" + esc(r.id) + '</td><td class="clip" title="' + esc(r.prompt) + '">' +
            esc(r.prompt) + "</td><td>" + r.best_score + "</td><td>" + r.queries + "</td><td>" +
            tag + "</td></tr>";
        }).join("");
        $("run_detail_table").innerHTML = '<table class="data"><thead><tr><th>ID</th><th>目标</th>' +
          "<th>最佳分</th><th>查询数</th><th>结果</th></tr></thead><tbody>" + rows + "</tbody></table>";
      }).catch(function () {});
  }

  $("btn_runs_refresh").addEventListener("click", loadRuns);

  /* ---------------- go ---------------- */
  initMeta();
})();
