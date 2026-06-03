from __future__ import annotations


GUI_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GemmAnima</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202529;
      --muted: #6c777d;
      --line: #d8dee2;
      --page: #eef2f3;
      --panel: #ffffff;
      --soft: #f6f8f9;
      --accent: #1f6f5b;
      --accent-strong: #165241;
      --accent-soft: #dcefe8;
      --user: #d9f0e7;
      --assistant: #ffffff;
      --warn-bg: #fff6e6;
      --warn-line: #e1bd83;
      --danger: #9a2935;
      --shadow: 0 14px 42px rgba(32, 37, 41, .08);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    button { border: 0; cursor: pointer; }
    .app-shell {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100vh;
      min-height: 0;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 18px;
      background: rgba(255, 255, 255, .94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .identity {
      display: flex;
      align-items: center;
      min-width: 0;
      gap: 11px;
    }
    .avatar, .mini-avatar {
      display: grid;
      place-items: center;
      border-radius: 50%;
      font-weight: 800;
    }
    .avatar {
      width: 38px;
      height: 38px;
      background: var(--accent);
      color: #fff;
      font-size: 16px;
    }
    .title-block { min-width: 0; }
    h1 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }
    .presence {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .presence-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #9aa4a9;
    }
    .presence-dot.ready { background: #1b9c64; }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .icon-button {
      display: grid;
      place-items: center;
      width: 40px;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: var(--soft);
      color: var(--ink);
      font-weight: 800;
    }
    .icon-button:hover { background: #edf4f1; }
    .chat-stage {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: 0;
    }
    .conversation {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 0;
    }
    .chat-log {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
      overflow: auto;
      padding: 22px min(7vw, 72px);
      scroll-behavior: smooth;
    }
    .message-row {
      display: flex;
      gap: 8px;
      max-width: min(760px, 100%);
      animation: rise .12s ease-out;
    }
    @keyframes rise {
      from { transform: translateY(4px); opacity: .7; }
      to { transform: translateY(0); opacity: 1; }
    }
    .message-row.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }
    .message-row.assistant, .message-row.system { align-self: flex-start; }
    .mini-avatar {
      flex: 0 0 28px;
      width: 28px;
      height: 28px;
      margin-top: 3px;
      background: #dfe7e9;
      color: #45545b;
      font-size: 12px;
    }
    .message-row.user .mini-avatar {
      background: var(--accent);
      color: #fff;
    }
    .bubble {
      max-width: min(680px, calc(100vw - 94px));
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--assistant);
      line-height: 1.52;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 1px 1px rgba(32, 37, 41, .03);
    }
    .user .bubble {
      background: var(--user);
      border-color: #bfddd2;
    }
    .system .bubble {
      background: var(--warn-bg);
      border-color: var(--warn-line);
      color: #684413;
    }
    .bubble p { margin: 0; }
    .meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .generated-image {
      display: block;
      width: min(100%, 640px);
      max-height: 68vh;
      object-fit: contain;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8faf9;
    }
    .artifact-link {
      display: inline-block;
      margin-top: 8px;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 750;
      text-decoration: none;
    }
    .artifact-link:hover { text-decoration: underline; }
    .generation-pending {
      display: grid;
      gap: 6px;
    }
    .generation-loading-line {
      display: flex;
      align-items: center;
      gap: 9px;
      font-weight: 750;
    }
    .generation-spinner {
      width: 18px;
      height: 18px;
      border: 3px solid #d7e4df;
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .generation-stage {
      color: var(--muted);
      font-size: 12px;
    }
    .composer-wrap {
      padding: 12px min(7vw, 72px) 16px;
      background: linear-gradient(180deg, rgba(238, 242, 243, 0), var(--page) 24%);
    }
    .composer {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: end;
      gap: 8px;
      max-width: 900px;
      margin: 0 auto;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .message-box {
      min-height: 42px;
      max-height: 160px;
      resize: none;
      border: 0;
      outline: 0;
      padding: 10px 4px;
      background: transparent;
      color: var(--ink);
      line-height: 1.45;
    }
    .send-button {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      background: var(--accent);
      color: #fff;
      font-weight: 900;
      font-size: 18px;
    }
    .send-button:hover { background: var(--accent-strong); }
    .send-button:disabled, .icon-button:disabled {
      opacity: .55;
      cursor: progress;
    }
    .attachment-strip {
      display: none;
      max-width: 900px;
      margin: 0 auto 8px;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .attachment-strip.visible { display: flex; }
    .attachment-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      max-width: 100%;
      padding: 7px 9px;
      border: 1px solid #c8ddd5;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      overflow: hidden;
    }
    .attachment-chip span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .clear-attachment {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: rgba(255,255,255,.75);
      color: var(--accent-strong);
    }
    .settings {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      padding: 16px;
      border-left: 1px solid var(--line);
      background: #f8faf9;
    }
    .settings.collapsed { display: none; }
    details {
      margin-bottom: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      padding: 11px 12px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 750;
    }
    .setting-body { padding: 0 12px 12px; }
    label {
      display: block;
      margin: 9px 0 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 9px;
    }
    .status-grid { display: grid; gap: 8px; }
    .status-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fff;
      font-size: 12px;
    }
    .status-item strong {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
    }
    .asset-actions {
      display: grid;
      gap: 8px;
    }
    .primary-action {
      width: 100%;
      padding: 9px 10px;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 800;
    }
    .primary-action:hover { background: var(--accent-strong); }
    .primary-action:disabled {
      opacity: .65;
      cursor: progress;
    }
    .progress-block {
      display: grid;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fff;
      font-size: 12px;
    }
    .progress-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
    }
    progress {
      width: 100%;
      height: 12px;
      accent-color: var(--accent);
    }
    .ok { color: var(--accent-strong); }
    .warn { color: #8a5b18; }
    .bad { color: var(--danger); }
    .path {
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 11px;
      line-height: 1.45;
    }
    .conflict-note {
      display: none;
      max-width: 900px;
      margin: 0 auto 8px;
      border: 1px solid var(--warn-line);
      border-radius: 8px;
      padding: 10px;
      background: var(--warn-bg);
      color: #684413;
      font-size: 13px;
    }
    .conflict-actions {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .conflict-field-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .conflict-field {
      min-width: 88px;
      font-weight: 750;
    }
    .conflict-action {
      width: auto;
      padding: 7px 9px;
      border: 1px solid var(--warn-line);
      border-radius: 6px;
      background: #f3ddc1;
      color: #5d3511;
      font-size: 12px;
    }
    .name-setup {
      position: fixed;
      inset: 0;
      z-index: 40;
      display: none;
      place-items: center;
      padding: 18px;
      background: rgba(18, 34, 42, .42);
      backdrop-filter: blur(4px);
    }
    .name-setup.visible { display: grid; }
    .name-card {
      width: min(420px, calc(100vw - 36px));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #fff;
      box-shadow: var(--shadow);
    }
    .name-card h2 {
      margin: 0 0 8px;
      font-size: 18px;
    }
    .name-card p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .name-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 12px;
    }
    .secondary-action {
      width: auto;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 750;
    }
    .drop-overlay {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: none;
      place-items: center;
      background: rgba(18, 34, 42, .34);
      backdrop-filter: blur(3px);
      pointer-events: none;
    }
    .drop-overlay.visible { display: grid; }
    .drop-card {
      display: grid;
      place-items: center;
      width: min(420px, calc(100vw - 40px));
      min-height: 170px;
      border: 2px dashed #b8d5cb;
      border-radius: 8px;
      background: #fff;
      color: var(--accent-strong);
      font-weight: 800;
      box-shadow: var(--shadow);
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0,0,0,0);
    }
    @media (max-width: 860px) {
      .chat-stage { grid-template-columns: 1fr; }
      .settings {
        position: fixed;
        inset: 62px 10px 10px;
        z-index: 10;
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
      }
      .chat-log { padding: 16px 12px; }
      .composer-wrap { padding: 10px 10px 14px; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="name-setup" class="name-setup" aria-modal="true" role="dialog" aria-labelledby="name-setup-title">
    <div class="name-card">
      <h2 id="name-setup-title">Name your chatbot</h2>
      <p>This name is stored in this browser only and can be changed later.</p>
      <label for="bot-name-input">Chatbot name</label>
      <input id="bot-name-input" type="text" maxlength="32" value="GemmAnima" autocomplete="off">
      <div class="name-actions">
        <button id="bot-name-save" class="primary-action" type="button">Start chat</button>
      </div>
    </div>
  </div>
  <div class="app-shell">
    <header class="topbar">
      <div class="identity">
        <div class="avatar" data-bot-name-target="avatar">G</div>
        <div class="title-block">
          <h1 data-bot-name-target="title">GemmAnima</h1>
          <div class="presence"><span id="presence-dot" class="presence-dot"></span><span id="top-status">연결 확인 중</span></div>
        </div>
      </div>
      <div class="top-actions">
        <button id="settings-toggle" class="icon-button" type="button" title="설정">⚙</button>
      </div>
    </header>

    <main class="chat-stage">
      <section class="conversation" aria-label="채팅">
        <div id="chat-log" class="chat-log">
          <div class="message-row assistant">
            <div class="mini-avatar" data-bot-name-target="avatar">G</div>
            <div>
              <div class="bubble">말을 걸거나 이미지를 요청해 주세요. 필요한 경우 첨부 이미지를 읽고, 그림 요청은 생성 단계로 넘깁니다.</div>
              <div class="meta" data-bot-name-target="meta">GemmAnima resident runtime</div>
            </div>
          </div>
        </div>
        <div class="composer-wrap">
          <div id="conflict-note" class="conflict-note"></div>
          <div id="attachment-strip" class="attachment-strip">
            <div class="attachment-chip">첨부 <span id="attachment-name"></span><button id="clear-attachment" class="clear-attachment" type="button" title="첨부 해제">×</button></div>
          </div>
          <div class="composer">
            <button id="attach-button" class="icon-button" type="button" title="이미지 첨부">＋</button>
            <label class="sr-only" for="message">메시지</label>
            <textarea id="message" class="message-box" rows="1" placeholder="메시지를 입력하세요."></textarea>
            <button id="send" class="send-button" type="button" title="보내기">›</button>
            <input id="file-input" type="file" accept="image/*" hidden>
            <input id="image_path" type="hidden">
          </div>
        </div>
      </section>

      <aside id="settings-panel" class="settings collapsed" aria-label="설정">
        <details open>
          <summary>생성 조건</summary>
          <div class="setting-body">
            <div class="row">
              <div>
                <label for="language">대화 언어</label>
                <select id="language">
                  <option value="ko">한국어</option>
                  <option value="en">English</option>
                </select>
              </div>
              <div>
                <label for="renderer">렌더러</label>
                <select id="renderer">
                  <option value="local-worker" selected>local-worker</option>
                  <option value="in-process">in-process</option>
                  <option value="dry-run">dry-run</option>
                  <option value="external-script">external-script</option>
                </select>
              </div>
            </div>
            <label for="generation_preset">생성 프리셋</label>
            <select id="generation_preset">
              <option value="anima_draft">Anima Draft</option>
              <option value="anima_balanced" selected>Anima Balanced</option>
              <option value="anima_final">Anima Final</option>
              <option value="anima_lora">Anima LoRA</option>
            </select>
            <div class="row">
              <div>
                <label for="resolution_preset">해상도</label>
                <select id="resolution_preset">
                  <option value="square_1024">1024 x 1024</option>
                  <option value="portrait_832_1216">832 x 1216</option>
                  <option value="portrait_768_1344">768 x 1344</option>
                  <option value="custom">커스텀</option>
                </select>
              </div>
              <div>
                <label for="orientation">방향</label>
                <select id="orientation">
                  <option value="portrait">세로</option>
                  <option value="landscape">가로</option>
                </select>
              </div>
            </div>
            <div class="row">
              <div>
                <label for="custom_width">너비</label>
                <input id="custom_width" type="number" min="256" max="2048" step="8" value="1024">
              </div>
              <div>
                <label for="custom_height">높이</label>
                <input id="custom_height" type="number" min="256" max="2048" step="8" value="1024">
              </div>
            </div>
            <div class="row">
              <div>
                <label for="steps">스텝</label>
                <input id="steps" type="number" min="1" max="80" value="">
              </div>
              <div>
                <label for="cfg">CFG</label>
                <input id="cfg" type="number" min="1" max="12" step="0.1" value="">
              </div>
            </div>
            <div class="row">
              <div>
                <label for="sampler">샘플러</label>
                <select id="sampler">
                  <option value="">프리셋 기본값</option>
                  <option value="euler">euler</option>
                  <option value="euler_ancestral">euler_ancestral</option>
                  <option value="dpmpp_2m">dpmpp_2m</option>
                  <option value="dpmpp_2m_sde_gpu">dpmpp_2m_sde_gpu</option>
                </select>
              </div>
              <div>
                <label for="scheduler">스케줄러</label>
                <select id="scheduler">
                  <option value="">프리셋 기본값</option>
                  <option value="normal">normal</option>
                  <option value="karras">karras</option>
                  <option value="sgm_uniform">sgm_uniform</option>
                </select>
              </div>
            </div>
            <div class="row">
              <div>
                <label for="seed">시드</label>
                <input id="seed" type="number" value="424242">
              </div>
              <div>
                <label for="dtype">UNet dtype</label>
                <select id="dtype">
                  <option value="fp8_e4m3fn_fast">fp8_e4m3fn_fast</option>
                  <option value="default">default</option>
                  <option value="fp8_e4m3fn">fp8_e4m3fn</option>
                  <option value="fp8_e5m2">fp8_e5m2</option>
                </select>
              </div>
            </div>
          </div>
        </details>

        <details>
          <summary>고급 라우팅</summary>
          <div class="setting-body">
            <label for="force_task">경로 강제 지정</label>
            <button id="bot-name-reset" class="secondary-action" type="button">Change chatbot name</button>
            <select id="force_task">
              <option value="">자동</option>
              <option value="chat">채팅</option>
              <option value="generate">이미지 생성</option>
              <option value="tag">이미지 태그</option>
            </select>
            <label for="force_chat_mode">출력 계약 강제 지정</label>
            <select id="force_chat_mode">
              <option value="">자동</option>
              <option value="general_chat">일반 대화</option>
              <option value="tag_request">태그 요청</option>
              <option value="image_generation_request">이미지 생성</option>
              <option value="status_question">상태 질문</option>
              <option value="file_checkpoint_question">파일/체크포인트 질문</option>
            </select>
            <label><input id="headroom_enabled" type="checkbox"> Headroom 컨텍스트 압축</label>
          </div>
        </details>

        <details>
          <summary>모델 자산</summary>
          <div class="setting-body asset-actions">
            <button id="model-download" class="primary-action" type="button">필요 모델 다운로드</button>
            <div class="progress-block">
              <div class="progress-label"><span>전체</span><span id="model-download-overall-label">대기</span></div>
              <progress id="model-download-overall" value="0" max="100"></progress>
              <div class="progress-label"><span>현재 파일</span><span id="model-download-file-label">대기</span></div>
              <progress id="model-download-file" value="0" max="100"></progress>
              <div id="model-download-current" class="path">다운로드가 필요하면 버튼을 누르세요.</div>
            </div>
          </div>
        </details>

        <details>
          <summary>상태</summary>
          <div class="setting-body">
            <div id="health" class="status-grid"></div>
          </div>
        </details>

        <details>
          <summary>디버그</summary>
          <div class="setting-body">
            <pre id="result">요청을 기다리는 중입니다.</pre>
          </div>
        </details>
      </aside>
    </main>
  </div>

  <div id="drop-overlay" class="drop-overlay">
    <div class="drop-card">이미지를 놓으면 대화에 첨부합니다</div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const conversationHistory = [];
    const DEFAULT_BOT_NAME = "GemmAnima";
    let currentConflict = null;
    let attachedImagePath = "";
    let attachedImageName = "";
    let dragDepth = 0;

    function normalizeBotName(value) {
      const cleaned = String(value || "").trim().replace(/\s+/g, " ");
      return cleaned.slice(0, 32) || DEFAULT_BOT_NAME;
    }

    function loadBotName() {
      return window.currentBotName || DEFAULT_BOT_NAME;
    }

    async function fetchBotName() {
      const res = await fetch("/v1/settings/chatbot-name");
      const data = await res.json();
      const name = normalizeBotName(data.chatbot_name);
      window.currentBotName = name;
      applyBotName(name);
      return data;
    }

    async function saveBotName(value) {
      const name = normalizeBotName(value);
      const res = await fetch("/v1/settings/chatbot-name", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({chatbot_name: name})
      });
      if (!res.ok) throw new Error("failed to save chatbot name");
      const data = await res.json();
      window.currentBotName = normalizeBotName(data.chatbot_name);
      applyBotName(name);
      return window.currentBotName;
    }

    function botInitial(name) {
      const trimmed = normalizeBotName(name);
      return trimmed.slice(0, 1).toUpperCase() || "G";
    }

    function applyBotName(name = loadBotName()) {
      const normalized = normalizeBotName(name);
      document.querySelectorAll('[data-bot-name-target="title"]').forEach((node) => {
        node.textContent = normalized;
      });
      document.querySelectorAll('[data-bot-name-target="avatar"]').forEach((node) => {
        node.textContent = botInitial(normalized);
      });
      document.querySelectorAll('[data-bot-name-target="meta"]').forEach((node) => {
        node.textContent = `${normalized} resident runtime`;
      });
      document.title = normalized;
      return normalized;
    }

    async function showNameSetupIfNeeded(force = false) {
      const setup = $("name-setup");
      if (!setup) return;
      const data = await fetchBotName().catch(() => ({chatbot_name: DEFAULT_BOT_NAME, configured: false}));
      const name = normalizeBotName(data.chatbot_name);
      const hasStoredName = Boolean(data.configured);
      $("bot-name-input").value = name;
      applyBotName(name);
      setup.classList.toggle("visible", force || !hasStoredName);
      if (force || !hasStoredName) $("bot-name-input").focus();
    }

    function hideNameSetup() {
      $("name-setup").classList.remove("visible");
    }

    function statusClass(value) {
      if (value === true || value === "ok" || value === "completed") return "ok";
      if (value === false || value === "failed") return "bad";
      return "warn";
    }

    function byteLabel(bytes) {
      const value = Number(bytes || 0);
      if (value >= 1024 * 1024 * 1024) return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
      if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
      if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
      return `${value} B`;
    }

    function updateDownloadGauge(data) {
      const total = Number(data.total_assets || 0);
      const done = Number(data.completed_assets || 0);
      const currentDownloaded = Number(data.current_downloaded_bytes || 0);
      const currentTotal = Number(data.current_total_bytes || 0);
      const filePercent = currentTotal > 0 ? Math.min(100, Math.round((currentDownloaded / currentTotal) * 100)) : 0;
      const overallPercent = total > 0
        ? Math.min(100, Math.round(((done + (filePercent / 100) * (data.status === "running" ? 1 : 0)) / total) * 100))
        : 0;
      $("model-download-overall").value = data.status === "completed" ? 100 : overallPercent;
      $("model-download-file").value = data.status === "completed" ? 100 : filePercent;
      $("model-download-overall-label").textContent = `${done}/${total || 0} · ${data.status || "idle"}`;
      $("model-download-file-label").textContent = currentTotal
        ? `${byteLabel(currentDownloaded)} / ${byteLabel(currentTotal)}`
        : data.status || "대기";
      $("model-download-current").textContent = data.error || data.current_asset || "다운로드가 필요하면 버튼을 누르세요.";
      $("model-download").disabled = Boolean(data.running || data.status === "running");
    }

    async function refreshDownloadStatus() {
      const res = await fetch("/v1/models/download/status");
      const data = await res.json();
      updateDownloadGauge(data);
      return data;
    }

    async function startModelDownload() {
      $("model-download").disabled = true;
      $("model-download-current").textContent = "다운로드 시작 중...";
      const res = await fetch("/v1/models/download", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({overwrite: false})
      });
      const data = await res.json();
      updateDownloadGauge(data);
      if (!res.ok || data.status === "failed") {
        throw new Error(data.error || "모델 다운로드 시작 실패");
      }
    }

    function nowLabel() {
      return new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    }

    function addBubble(role, text, meta = "") {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;
      const avatar = document.createElement("div");
      avatar.className = "mini-avatar";
      avatar.textContent = role === "user" ? "U" : role === "system" ? "!" : botInitial(loadBotName());
      const stack = document.createElement("div");
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (text) bubble.textContent = text;
      stack.appendChild(bubble);
      const metaNode = document.createElement("div");
      metaNode.className = "meta";
      metaNode.textContent = meta || nowLabel();
      stack.appendChild(metaNode);
      row.appendChild(avatar);
      row.appendChild(stack);
      $("chat-log").appendChild(row);
      $("chat-log").scrollTop = $("chat-log").scrollHeight;
      return bubble;
    }

    function autoGrowMessageBox() {
      const box = $("message");
      box.style.height = "auto";
      box.style.height = `${Math.min(box.scrollHeight, 160)}px`;
    }

    function responseSummary(data) {
      if (data.error) return `오류: ${data.error}`;
      if (data.mode === "generate_image" && isImagePath(data.output_path)) {
        return data.message || "이미지를 만들었습니다.";
      }
      const message = data.message || data.tags || "완료했습니다.";
      const details = [];
      if (data.output_path) {
        const label = String(data.output_path).endsWith(".dryrun.txt") ? "dry-run report" : "output";
        details.push(`${label}: ${data.output_path}`);
      }
      if (data.manifest_path) details.push(`manifest: ${data.manifest_path}`);
      if (data.mode && data.mode !== "chat") details.push(`mode: ${data.mode}`);
      return details.length ? `${message}\n\n${details.join("\n")}` : message;
    }

    function isImagePath(path) {
      return /\.(png|jpe?g|webp|gif|bmp)$/i.test(String(path || ""));
    }

    function imageArtifactPath(outputPath) {
      const normalized = String(outputPath || "").replaceAll("\\", "/");
      const lower = normalized.toLowerCase();
      const imagesMarker = "/images/";
      if (lower.startsWith("runs/images/")) return normalized.slice("runs/images/".length);
      const markerIndex = lower.lastIndexOf(imagesMarker);
      if (markerIndex >= 0) return normalized.slice(markerIndex + imagesMarker.length);
      return normalized.split("/").pop();
    }

    function encodeArtifactPath(path) {
      return String(path || "").split("/").filter(Boolean).map((part) => encodeURIComponent(part)).join("/");
    }

    function artifactUrl(outputPath) {
      return `/artifacts/images/${encodeArtifactPath(imageArtifactPath(outputPath))}`;
    }

    function isLikelyGenerationRequest(message, payload) {
      if (payload.task === "generate" || payload.chat_mode === "image_generation_request") return true;
      const text = String(message || "").toLowerCase();
      const metaQuestionWords = ["어떻게", "무엇", "구분", "품질", "설명", "규칙", "라우팅", "프리셋", "why", "how", "what"];
      if (metaQuestionWords.some((word) => text.includes(word))) return false;
      const generationWords = [
        "이미지를 만들어줘",
        "이미지 만들어줘",
        "이미지를 생성해",
        "이미지 생성",
        "그림 그려",
        "그림을 그려",
        "그림 만들어",
        "일러스트",
        "draw",
        "render",
        "generate image",
        "generate an image",
        "create image",
        "create an image",
        "anime illustration"
      ];
      if (generationWords.some((word) => text.includes(word))) return true;
      if (/(이미지|그림).*(만들|생성|그려|그리|구성|새로)/i.test(text)) return true;
      if (/(이미지\s*요청이\s*아닌|이미지\s*생성이\s*아닌|그림\s*요청이\s*아닌|not image|not an image)/i.test(text)) {
        return false;
      }
      if (/(어떻게|왜|무엇|구분|품질|설명|규칙|라우팅|프리셋)/i.test(text)) {
        return false;
      }
      return /(이미지(을|를)?\s*(만들어|생성해|그려)\s*(줘|주|주세요)?|그림(을|를)?\s*(그려|만들어|생성해)\s*(줘|주|주세요)?|일러스트(를)?\s*(그려|만들어|생성해)\s*(줘|주|주세요)?|draw|render\s+.+|generate\s+(an\s+)?image|create\s+(an\s+)?image|anime illustration)/i.test(text);
    }

    function updatePendingGenerationStage(pending, stage) {
      if (!pending || !pending.stageNode) return;
      pending.stageNode.textContent = stage;
    }

    function renderPendingGeneration(payload) {
      const stages = [
        "요청 해석 중",
        "생성 프리셋 적용 중",
        `${payload.resolution_preset || "resolution"} · ${payload.sampler || "preset sampler"} · ${payload.scheduler || "preset scheduler"}`,
        "Anima 렌더러 준비 중",
        "이미지 생성 중",
        "결과를 채팅에 첨부하는 중"
      ];
      const row = document.createElement("div");
      row.className = "message-row assistant generation-pending-row";
      const avatar = document.createElement("div");
      avatar.className = "mini-avatar";
      avatar.textContent = botInitial(loadBotName());
      const stack = document.createElement("div");
      const bubble = document.createElement("div");
      bubble.className = "bubble generation-pending";
      const line = document.createElement("div");
      line.className = "generation-loading-line";
      const spinner = document.createElement("span");
      spinner.className = "generation-spinner";
      spinner.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.textContent = "이미지 생성 중";
      line.appendChild(spinner);
      line.appendChild(label);
      const stage = document.createElement("div");
      stage.className = "generation-stage";
      stage.textContent = stages[0];
      bubble.appendChild(line);
      bubble.appendChild(stage);
      const metaNode = document.createElement("div");
      metaNode.className = "meta";
      metaNode.textContent = nowLabel();
      stack.appendChild(bubble);
      stack.appendChild(metaNode);
      row.appendChild(avatar);
      row.appendChild(stack);
      $("chat-log").appendChild(row);
      $("chat-log").scrollTop = $("chat-log").scrollHeight;
      let index = 0;
      const timer = setInterval(() => {
        index = Math.min(index + 1, stages.length - 1);
        updatePendingGenerationStage({stageNode: stage}, stages[index]);
      }, 2400);
      return {row, bubble, labelNode: label, stageNode: stage, timer};
    }

    function renderThinkingBubble(payload) {
      const row = document.createElement("div");
      row.className = "message-row assistant generation-pending-row";
      const avatar = document.createElement("div");
      avatar.className = "mini-avatar";
      avatar.textContent = botInitial(loadBotName());
      const stack = document.createElement("div");
      const bubble = document.createElement("div");
      bubble.className = "bubble generation-pending";
      const line = document.createElement("div");
      line.className = "generation-loading-line";
      const spinner = document.createElement("span");
      spinner.className = "generation-spinner";
      spinner.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.textContent = "생각 중...";
      line.appendChild(spinner);
      line.appendChild(label);
      const stage = document.createElement("div");
      stage.className = "generation-stage";
      stage.textContent = "의도 판단 중";
      bubble.appendChild(line);
      bubble.appendChild(stage);
      const metaNode = document.createElement("div");
      metaNode.className = "meta";
      metaNode.textContent = nowLabel();
      stack.appendChild(bubble);
      stack.appendChild(metaNode);
      row.appendChild(avatar);
      row.appendChild(stack);
      $("chat-log").appendChild(row);
      $("chat-log").scrollTop = $("chat-log").scrollHeight;
      return {row, bubble, labelNode: label, stageNode: stage, timer: null};
    }

    function updatePendingFromStreamEvent(pending, event) {
      if (!pending || !event) return;
      const stage = String(event.stage || "");
      if (event.message && pending.stageNode) pending.stageNode.textContent = event.message;
      if (!pending.labelNode) return;
      if (stage === "thinking") pending.labelNode.textContent = "생각 중...";
      else if (stage === "routing") pending.labelNode.textContent = "판단 중...";
      else if (stage === "generating") pending.labelNode.textContent = "이미지 생성 중...";
      else if (stage === "tagging") pending.labelNode.textContent = "이미지 분석 중...";
      else if (stage === "chatting") pending.labelNode.textContent = "답변 작성 중...";
      else if (stage === "error") pending.labelNode.textContent = "오류";
    }

    function replacePendingGeneration(pending, data) {
      if (!pending) return null;
      if (pending.timer) clearInterval(pending.timer);
      pending.bubble.className = "bubble";
      pending.bubble.replaceChildren();
      pending.row.className = `message-row ${data.error ? "system" : "assistant"}`;
      return pending.bubble;
    }

    function addAssistantResponse(data, targetBubble = null) {
      const bubble = targetBubble || addBubble(data.error ? "system" : "assistant", "");
      const summary = document.createElement("p");
      summary.textContent = responseSummary(data);
      bubble.appendChild(summary);
      if (!data.error && isImagePath(data.output_path)) {
        const url = artifactUrl(data.output_path);
        const img = document.createElement("img");
        img.className = "generated-image";
        img.src = url;
        img.alt = "생성 이미지";
        img.loading = "lazy";
        bubble.appendChild(img);
        const link = document.createElement("a");
        link.className = "artifact-link";
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = "이미지 열기";
        bubble.appendChild(link);
      }
      $("chat-log").scrollTop = $("chat-log").scrollHeight;
    }

    function setAttachment(path, name) {
      attachedImagePath = path || "";
      attachedImageName = name || path || "";
      $("image_path").value = attachedImagePath;
      $("attachment-name").textContent = attachedImageName;
      $("attachment-strip").classList.toggle("visible", Boolean(attachedImagePath));
    }

    async function fileToDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    async function uploadImageFile(file) {
      if (!file || !file.type.startsWith("image/")) {
        addBubble("system", "이미지 파일만 첨부할 수 있습니다.");
        return;
      }
      addBubble("system", `${file.name} 업로드 중...`);
      const dataUrl = await fileToDataUrl(file);
      const res = await fetch("/v1/uploads", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({file_name: file.name, data_url: dataUrl})
      });
      const data = await res.json();
      if (!res.ok || data.status !== "completed") {
        throw new Error(data.error || "업로드 실패");
      }
      setAttachment(data.path, file.name);
      addBubble("system", `${file.name} 첨부 완료`);
    }

    async function runStreamingRequest(payload, pending) {
      const res = await fetch("/v1/chat/stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      if (!res.ok || !res.body) {
        const fallback = await fetch("/v1/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        return await fallback.json();
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === "complete") return event.data;
          if (event.type === "error") throw new Error(event.error || event.message || "stream failed");
          updatePendingFromStreamEvent(pending, event);
        }
      }
      if (buffer.trim()) {
        const event = JSON.parse(buffer);
        if (event.type === "complete") return event.data;
        if (event.type === "error") throw new Error(event.error || event.message || "stream failed");
      }
      throw new Error("stream ended without a result");
    }

    function handleDroppedFiles(files) {
      const file = Array.from(files || []).find((item) => item.type && item.type.startsWith("image/"));
      if (!file) {
        addBubble("system", "드롭한 항목에서 이미지 파일을 찾지 못했습니다.");
        return;
      }
      uploadImageFile(file).catch((err) => addBubble("system", String(err)));
    }

    async function refreshHealth() {
      const res = await fetch("/v1/health");
      const data = await res.json();
      const runtime = data.tipo_text?.resident_runtime || {};
      const ready = data.status === "ok" && runtime.initialized;
      $("presence-dot").classList.toggle("ready", Boolean(ready));
      $("top-status").textContent = ready ? "Gemma 상주 중" : "상태 확인 필요";
      const items = [];
      items.push(`<div class="status-item"><strong>Gemma runtime</strong><span class="${statusClass(runtime.status)}">${runtime.initialized ? "상주" : "대기"}</span><div class="path">${runtime.model || ""}</div></div>`);
      for (const [name, item] of Object.entries(data.models || {})) {
        items.push(`<div class="status-item"><strong>${name}</strong><span class="${statusClass(item.exists)}">${item.exists ? "준비됨" : "누락"}</span><div class="path">${item.path}</div></div>`);
      }
      $("health").innerHTML = items.join("");
      refreshDownloadStatus().catch(() => {});
    }

    function shouldTagAttachedImage(message) {
      return attachedImagePath && /(태그|tag|분석|설명|describe|caption)/i.test(message);
    }

    function shouldTagThenGenerateAttachedImage(message) {
      if (!attachedImagePath) return false;
      const text = String(message || "").toLowerCase();
      const hasTag = /tag|describe|caption/i.test(text) || text.includes("\ud0dc\uadf8") || text.includes("\ud0dc\uae45") || text.includes("\ubd84\uc11d") || text.includes("\uc124\uba85");
      const hasGenerate = /draw|render|generate|create|image/i.test(text) || text.includes("\uc774\ubbf8\uc9c0") || text.includes("\uadf8\ub9bc") || text.includes("\uc0dd\uc131") || text.includes("\uadf8\ub824") || text.includes("\ub9cc\ub4e4");
      const hasSequence = /then|after|from those|with those/i.test(text) || text.includes("\ud6c4") || text.includes("\ub4a4") || text.includes("\uae30\ubc18") || text.includes("\ubc14\ud0d5") || text.includes("\uadf8 \ud0dc\uadf8");
      return hasTag && hasGenerate && hasSequence;
    }

    async function runRequest() {
      const message = $("message").value.trim();
      if (!message && !attachedImagePath) return;
      $("send").disabled = true;
      $("result").textContent = "실행 중...";
      const displayText = attachedImageName ? `${message || "첨부 이미지"}\n[첨부: ${attachedImageName}]` : message;
      addBubble("user", displayText);
      const forcedTask = $("force_task").value;
      const payload = {
        task: $("force_task").value || "auto",
        message: message || "이 이미지를 태그로 설명해줘.",
        language: $("language").value,
        image_path: attachedImagePath,
        history: conversationHistory.slice(-16),
        renderer: $("renderer").value,
        generation_preset: $("generation_preset").value,
        resolution_preset: $("resolution_preset").value,
        orientation: $("orientation").value,
        custom_width: $("custom_width").value,
        custom_height: $("custom_height").value,
        steps: $("steps").value,
        cfg: $("cfg").value,
        sampler: $("sampler").value,
        scheduler: $("scheduler").value,
        seed: $("seed").value,
        unet_dtype: $("dtype").value,
        headroom_enabled: $("headroom_enabled").checked
      };
      if (attachedImagePath) payload.reference_image_path = attachedImagePath;
      if (!forcedTask && shouldTagThenGenerateAttachedImage(message)) payload.task = "tag_then_generate";
      else if (!forcedTask && shouldTagAttachedImage(message)) payload.task = "tag";
      const forcedChatMode = $("force_chat_mode").value;
      if (forcedChatMode) payload.chat_mode = forcedChatMode;
      const pending = renderThinkingBubble(payload);
      try {
        const data = await runStreamingRequest(payload, pending);
        conversationHistory.push({role: "user", content: displayText});
        if (data.message) conversationHistory.push({role: "assistant", content: data.message});
        renderResult(data, pending);
        $("message").value = "";
        autoGrowMessageBox();
        await refreshHealth();
      } catch (err) {
        $("result").textContent = String(err);
        const targetBubble = replacePendingGeneration(pending, {error: String(err)});
        if (targetBubble) addAssistantResponse({error: String(err)}, targetBubble);
        else addBubble("system", String(err));
      } finally {
        $("send").disabled = false;
      }
    }

    function fieldLabel(field) {
      return String(field || "conflict").replaceAll("_", " ");
    }

    function conflictItems(conflict) {
      return Array.isArray(conflict?.conflicts) ? conflict.conflicts : [];
    }

    function conflictFields(conflict) {
      const fields = [];
      for (const item of conflictItems(conflict)) if (item?.field) fields.push(item.field);
      if (Array.isArray(conflict?.fields)) for (const field of conflict.fields) if (field) fields.push(field);
      return [...new Set(fields)].length ? [...new Set(fields)] : ["conflict"];
    }

    function conflictItemForField(conflict, field) {
      return conflictItems(conflict).find((item) => item?.field === field) || {};
    }

    function clarificationMessage(action, field, item) {
      const label = fieldLabel(field);
      if (action === "preserve") return `참조의 ${label} 유지.`;
      const requested = item?.text_value ? `: ${item.text_value}` : "";
      return `${label}는 요청 반영${requested}.`;
    }

    function sendClarification(field, action) {
      const conflict = currentConflict || {};
      const item = conflictItemForField(conflict, field);
      $("message").value = clarificationMessage(action, field, item);
      return runRequest();
    }

    function renderResult(data, pendingGeneration = null) {
      const note = $("conflict-note");
      note.replaceChildren();
      if (data.clarification_required && data.conflict) {
        currentConflict = data.conflict;
        const fields = conflictFields(data.conflict);
        note.style.display = "block";
        const summary = document.createElement("div");
        summary.textContent = `충돌: ${fields.join(", ")}. ${data.message || ""}`;
        note.appendChild(summary);
        const actions = document.createElement("div");
        actions.className = "conflict-actions";
        for (const field of fields) {
          const row = document.createElement("div");
          row.className = "conflict-field-actions";
          const label = document.createElement("span");
          label.className = "conflict-field";
          label.textContent = fieldLabel(field);
          row.appendChild(label);
          for (const [action, text] of [["preserve", "참조 유지"], ["change", "요청 반영"]]) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "conflict-action";
            button.setAttribute("data-conflict-action", action);
            button.setAttribute("data-conflict-field", field);
            button.textContent = text;
            row.appendChild(button);
          }
          actions.appendChild(row);
        }
        note.appendChild(actions);
      } else {
        currentConflict = null;
        note.style.display = "none";
      }
      $("result").textContent = JSON.stringify(data, null, 2);
      addAssistantResponse(data, replacePendingGeneration(pendingGeneration, data));
    }

    window.GemmAnimaTest = {
      async send(message, overrides = {}) {
        for (const [id, value] of Object.entries(overrides || {})) {
          const field = $(id);
          if (field) field.value = value;
        }
        $("message").value = String(message || "");
        autoGrowMessageBox();
        await runRequest();
        return this.stats();
      },
      attachImage(path, name = "attached-image.png") {
        setAttachment(String(path || ""), String(name || path || ""));
        return this.stats();
      },
      clearAttachment() {
        setAttachment("", "");
        return this.stats();
      },
      stats() {
        const rows = Array.from(document.querySelectorAll(".message-row"));
        return {
          rows: rows.length,
          userRows: document.querySelectorAll(".message-row.user").length,
          assistantRows: document.querySelectorAll(".message-row.assistant").length,
          systemRows: document.querySelectorAll(".message-row.system").length,
          attachedImagePath,
          attachedImageName,
          images: Array.from(document.querySelectorAll(".generated-image")).map((img) => img.src),
          links: Array.from(document.querySelectorAll(".artifact-link")).map((link) => link.href),
          lastTexts: rows.slice(-6).map((row) => row.textContent.trim()),
          result: $("result").textContent
        };
      }
    };

    async function runBrowserAutotest() {
      const params = new URLSearchParams(window.location.search);
      const requested = Number(params.get("autotest") || 0);
      if (!requested) return;
      const turns = Math.max(1, Math.min(40, requested));
      const messages = [
        "안녕. 브라우저 대화 테스트를 시작하자.",
        "지금 모델 런타임 상태를 짧게 설명해줘.",
        "한국어로 계속 대화해줘.",
        "이미지 요청이 아닌 일반 잡담으로 응답해줘.",
        "이 앱에서 중요한 기능을 한 문장으로 요약해줘.",
        "답변은 너무 길지 않게 유지해줘.",
        "태그 요청이 오면 영어 태그로 출력해야 한다는 규칙을 기억해줘.",
        "사용자가 한국어로 말해도 생성 태그는 한국어로 번역하면 안 되지?",
        "지금까지 대화 맥락을 짧게 이어받아줘.",
        "모델 다운로드 게이지가 왜 필요한지 설명해줘.",
        "일반 채팅과 이미지 생성 요청을 어떻게 구분해야 해?",
        "프롬프트 보강은 언제 해야 할까?",
        "이미지 품질을 올리려면 어떤 요소를 봐야 해?",
        "브릿지 어댑터가 하는 일을 한 줄로 말해줘.",
        "1024x1024, 30 steps, euler ancestral, sgm uniform으로 forest light anime girl 이미지를 만들어줘.",
        "이미지 생성 뒤에도 대화가 계속 되는지 확인하자.",
        "방금 요청은 이미지 생성 경로였고, 지금은 다시 채팅 경로야.",
        "사용자 의도 분류가 잘못되면 어떤 문제가 생겨?",
        "대화 중에 안전하게 실패를 알려주는 방식은?",
        "프리셋은 왜 사전에 정해야 해?",
        "해상도 프리셋 3개를 기억해?",
        "가로세로 전환은 기본 기능이어야 하지?",
        "드래그 앤 드롭 첨부 기능은 어디에 쓰일까?",
        "이미지 태그 요청과 생성 요청을 헷갈리지 말아줘.",
        "지금 답변은 자연스럽고 짧게.",
        "로컬 앱이 외부 API에 의존하면 안 되는 이유는?",
        "Gemma 모델이 VRAM에 상주해야 하는 이유는?",
        "LoRA와 mmproj는 필요할 때 붙이는 구조가 맞지?",
        "이미지 생성 결과가 채팅에 보여야 하는 이유는?",
        "작은 아이콘 중심 UI가 왜 좋은지 말해줘.",
        "이번에는 dry-run이 아니라 실제 생성 결과가 있으면 이미지가 채팅에 떠야 해.",
        "모델 카드에는 어떤 정보를 넣어야 해?",
        "라이선스 고지는 왜 중요해?",
        "프로토타입이라고 명시하는 게 맞지?",
        "GitHub에는 모델 파일을 올리면 안 되지?",
        "첫 실행 자동 다운로드는 어떤 장점이 있어?",
        "진행률 게이지는 사용자 불안을 줄여주지?",
        "마지막에서 두 번째 테스트 대화야.",
        "40턴 브라우저 테스트를 마무리하면서 이상 징후를 요약해줘.",
        "최종 턴이야. 대화와 이미지 생성이 이어졌는지 짧게 답해줘."
      ];
      const status = document.createElement("div");
      status.id = "autotest-status";
      status.className = "path";
      status.textContent = "autotest starting";
      $("chat-log").appendChild(status);
      for (let index = 0; index < turns; index += 1) {
        const isImageTurn = index === 14;
        $("force_task").value = "";
        $("force_chat_mode").value = "";
        if (isImageTurn) {
          $("renderer").value = "local-worker";
          $("resolution_preset").value = "square_1024";
          $("orientation").value = "portrait";
          $("steps").value = "30";
          $("sampler").value = "euler_ancestral";
          $("scheduler").value = "sgm_uniform";
          $("cfg").value = "5.0";
        }
        status.textContent = `autotest ${index + 1}/${turns}`;
        $("message").value = messages[index] || `테스트 대화 ${index + 1}`;
        autoGrowMessageBox();
        await runRequest();
      }
      $("force_task").value = "";
      status.textContent = `autotest completed ${turns}/${turns}`;
      status.dataset.completed = "true";
      status.dataset.turns = String(turns);
      status.dataset.images = String(document.querySelectorAll(".generated-image").length);
      status.dataset.rows = String(document.querySelectorAll(".message-row").length);
    }

    $("send").addEventListener("click", runRequest);
    $("attach-button").addEventListener("click", () => $("file-input").click());
    $("file-input").addEventListener("change", (event) => handleDroppedFiles(event.target.files));
    $("clear-attachment").addEventListener("click", () => setAttachment("", ""));
    $("settings-toggle").addEventListener("click", () => $("settings-panel").classList.toggle("collapsed"));
    $("bot-name-save").addEventListener("click", async () => {
      try {
        await saveBotName($("bot-name-input").value);
        hideNameSetup();
        $("message").focus();
      } catch (err) {
        addBubble("system", String(err));
      }
    });
    $("bot-name-reset").addEventListener("click", () => {
      showNameSetupIfNeeded(true).catch((err) => addBubble("system", String(err)));
    });
    $("bot-name-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        $("bot-name-save").click();
      }
    });
    $("model-download").addEventListener("click", () => {
      startModelDownload()
        .then(() => addBubble("system", "모델 다운로드를 시작했습니다. 설정 패널에서 진행률을 볼 수 있습니다."))
        .catch((err) => addBubble("system", String(err)));
    });
    $("message").addEventListener("input", autoGrowMessageBox);
    $("message").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        runRequest();
      }
    });
    $("conflict-note").addEventListener("click", (event) => {
      const button = event.target.closest("[data-conflict-action]");
      if (!button) return;
      sendClarification(button.dataset.conflictField, button.dataset.conflictAction);
    });
    window.addEventListener("dragenter", (event) => {
      event.preventDefault();
      dragDepth += 1;
      $("drop-overlay").classList.add("visible");
    });
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("dragleave", (event) => {
      event.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) $("drop-overlay").classList.remove("visible");
    });
    window.addEventListener("drop", (event) => {
      event.preventDefault();
      dragDepth = 0;
      $("drop-overlay").classList.remove("visible");
      handleDroppedFiles(event.dataTransfer.files);
    });
    refreshHealth().catch((err) => {
      $("top-status").textContent = "상태 확인 필요";
      $("result").textContent = String(err);
      addBubble("system", String(err));
    });
    showNameSetupIfNeeded().catch((err) => addBubble("system", String(err)));
    setInterval(() => {
      refreshDownloadStatus().catch(() => {});
    }, 1000);
    runBrowserAutotest().catch((err) => addBubble("system", String(err)));
  </script>
</body>
</html>
"""
