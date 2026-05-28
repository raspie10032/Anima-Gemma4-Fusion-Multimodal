from __future__ import annotations


GUI_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GemmAnima Console</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1d2528;
      --muted: #5d6b70;
      --line: #d8e0dc;
      --panel: #ffffff;
      --surface: #f4f7f4;
      --accent: #27745e;
      --accent-strong: #185241;
      --warn: #a15c1b;
      --bad: #9f2d38;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--surface);
      color: var(--ink);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: #eef5f1;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      padding: 16px;
      border-right: 1px solid var(--line);
      background: #f9fbf8;
    }
    section {
      padding: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 14px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 10px 0 5px;
    }
    input, select, textarea, button {
      width: 100%;
      font: inherit;
    }
    input, select, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
    }
    textarea {
      min-height: 132px;
      resize: vertical;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .55; cursor: progress; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }
    .status-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .status-item strong {
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .ok { color: var(--accent-strong); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    pre {
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
    }
    .output {
      min-height: 260px;
    }
    .path {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>GemmAnima Console</h1>
    <div id="top-status" class="path">loading</div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Generate</h2>
        <label for="message">Request</label>
        <textarea id="message">draw Nahida from Genshin Impact, gentle forest light, anime illustration</textarea>
        <label for="renderer">Renderer</label>
        <select id="renderer">
          <option value="dry-run">dry-run</option>
          <option value="in-process">in-process</option>
          <option value="external-script">external-script</option>
        </select>
        <div class="row">
          <div>
            <label for="steps">Steps</label>
            <input id="steps" type="number" min="1" max="60" value="8">
          </div>
          <div>
            <label for="size">Size</label>
            <input id="size" type="number" min="256" max="1536" step="64" value="512">
          </div>
        </div>
        <div class="row">
          <div>
            <label for="cfg">CFG</label>
            <input id="cfg" type="number" min="1" max="12" step="0.1" value="4.5">
          </div>
          <div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" value="424242">
          </div>
        </div>
        <label for="dtype">UNet dtype</label>
        <select id="dtype">
          <option value="fp8_e4m3fn_fast">fp8_e4m3fn_fast</option>
          <option value="default">default</option>
          <option value="fp8_e4m3fn">fp8_e4m3fn</option>
          <option value="fp8_e5m2">fp8_e5m2</option>
        </select>
        <button id="send" style="margin-top: 14px;">Run</button>
      </div>
    </aside>
    <section>
      <div class="panel">
        <h2>Health</h2>
        <div id="health" class="status-grid"></div>
      </div>
      <div class="panel output">
        <h2>Result</h2>
        <pre id="result">Waiting for a request.</pre>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);

    function statusClass(value) {
      if (value === true || value === "ok") return "ok";
      if (value === false) return "bad";
      return "warn";
    }

    async function refreshHealth() {
      const res = await fetch("/v1/health");
      const data = await res.json();
      $("top-status").textContent = data.status === "ok" ? "backend online" : "backend issue";
      const items = [];
      for (const [name, item] of Object.entries(data.models || {})) {
        items.push(`<div class="status-item"><strong>${name}</strong><span class="${statusClass(item.exists)}">${item.exists ? "ready" : "missing"}</span><div class="path">${item.path}</div></div>`);
      }
      const bridge = data.hiddenstage_bridge || {};
      items.push(`<div class="status-item"><strong>hiddenstage bridge</strong><span class="${statusClass(bridge.passed_mse_gate)}">${bridge.passed_mse_gate ? "passed" : "check"}</span><div class="path">val_mse: ${bridge.val_mse ?? "n/a"}</div></div>`);
      for (const [name, item] of Object.entries(data.renderers || {})) {
        items.push(`<div class="status-item"><strong>${name}</strong><span class="${statusClass(item.ready)}">${item.ready ? "ready" : "pending"}</span><div class="path">${Object.keys(item.checks || {}).length} checks</div></div>`);
      }
      $("health").innerHTML = items.join("");
    }

    async function runRequest() {
      $("send").disabled = true;
      $("result").textContent = "Running...";
      const payload = {
        message: $("message").value,
        renderer: $("renderer").value,
        steps: Number($("steps").value),
        size: Number($("size").value),
        cfg: Number($("cfg").value),
        seed: $("seed").value,
        unet_dtype: $("dtype").value
      };
      try {
        const res = await fetch("/v1/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        $("result").textContent = JSON.stringify(data, null, 2);
        await refreshHealth();
      } catch (err) {
        $("result").textContent = String(err);
      } finally {
        $("send").disabled = false;
      }
    }

    $("send").addEventListener("click", runRequest);
    refreshHealth().catch((err) => {
      $("top-status").textContent = "backend issue";
      $("result").textContent = String(err);
    });
  </script>
</body>
</html>
"""
