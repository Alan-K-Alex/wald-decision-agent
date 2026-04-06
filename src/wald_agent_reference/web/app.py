from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from ..chat import ChatManager, ChatSettingsFactory
from ..core.agent import LeadershipInsightAgent
from ..core.config import AppSettings, load_settings
from ..core.logging import configure_logging, get_logger
from ..memory.memory_backends import build_memory_backend
from ..utils import slugify


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    configure_logging(app_settings)
    logger = get_logger("web.app")
    chat_manager = ChatManager(app_settings)
    settings_factory = ChatSettingsFactory(app_settings)

    app = FastAPI(title="Wald Agent Reference")
    app.mount("/artifacts", StaticFiles(directory=app_settings.chats_path, check_dir=False), name="artifacts")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/chats")
    def list_chats() -> list[dict]:
        logger.info("Listing chats")
        return chat_manager.list_chats()

    @app.post("/api/chats")
    def create_chat(title: Optional[str] = Form(default=None)) -> dict:
        session = chat_manager.create_chat(title=title)
        logger.info("Created chat via API: %s", session.chat_id)
        return {"chat_id": session.chat_id, "title": session.title}

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict:
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        logger.info("Loaded chat via API: %s", chat_id)
        return {
            "chat_id": session.chat_id,
            "title": session.title,
            "messages": chat_manager.load_messages(session),
            "doc_count": sum(1 for path in session.docs_dir.rglob("*") if path.is_file()),
            "db_path": str(session.db_path),
        }

    @app.post("/api/chats/{chat_id}/upload")
    async def upload_folder(
        chat_id: str,
        files: List[UploadFile] = File(...),
        relative_paths: Optional[List[str]] = Form(default=None),
        incremental: bool = Form(default=True),
    ) -> dict:
        """
        Upload files to a chat.
        
        Args:
            chat_id: The chat ID
            files: Files to upload
            relative_paths: Optional relative paths for files
            incremental: If True (default), append to existing files. If False, replace all files and clear chat history.
        """
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc

        logger.info("Received upload request for chat %s with %d files (incremental=%s)", chat_id, len(files), incremental)
        payload: list[tuple[str, bytes]] = []
        for index, upload in enumerate(files):
            relative_path = (
                relative_paths[index]
                if relative_paths is not None and index < len(relative_paths) and relative_paths[index]
                else upload.filename
            )
            payload.append((relative_path, await upload.read()))
        counts = chat_manager.upload_files(session, payload, incremental=incremental)

        agent = LeadershipInsightAgent(settings_factory.build(session))
        summary = agent.prepare_documents(session.docs_dir)
        logger.info(
            "Completed upload ingestion for chat %s | files=%d documents=%d tables=%d visuals=%d",
            chat_id,
            counts["file_count"],
            summary["documents"],
            summary["tables"],
            summary["visuals"],
        )
        
        return {"chat_id": chat_id, **counts, **summary, "incremental": incremental}
        return {"chat_id": chat_id, **counts, **summary}

    @app.post("/api/chats/{chat_id}/ask")
    def ask_chat(chat_id: str, question: str = Form(...), generate_plot: bool = Form(default=True)) -> dict:
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        if not session.docs_dir.exists() or not any(path.is_file() for path in session.docs_dir.rglob("*")):
            logger.warning("Ask attempted before upload for chat %s", chat_id)
            raise HTTPException(status_code=400, detail="Upload a folder before asking questions")

        logger.info("Received question for chat %s: %s", chat_id, question)
        chat_settings = settings_factory.build(session)
        agent = LeadershipInsightAgent(chat_settings)
        response = agent.ask(question=question, docs_path=session.docs_dir, generate_plot=generate_plot)
        chat_manager.record_exchange(session, question, response)
        
        # Reload session to get updated title
        updated_session = chat_manager.load_chat(chat_id)
        logger.info("Answered question for chat %s", chat_id)
        
        # Convert to concise chat response (title + findings + plots only)
        # Full details are in the PDF report
        chat_response = response.to_chat_response()
        
        return {
            "chat_id": chat_id,
            "title": updated_session.title,  # DYNAMICALLY UPDATED TITLE
            "question": question,
            "answer": chat_response.answer,
            "key_findings": chat_response.key_findings,
            "visual_insights": chat_response.visual_insights,
            "plot_urls": [f"/artifacts/{chat_id}/artifacts/plots/{Path(path).name}" for path in response.plot_paths],
            "plot_base64": chat_response.plots_base64[0] if chat_response.plots_base64 else "",
            "source_summary": chat_response.source_summary,
            "data_types_used": chat_response.data_types_used,
            "report_url": f"/artifacts/{chat_id}/artifacts/reports/{slugify(question)}.md",
        }

    @app.post("/api/chats/{chat_id}/ask-compact")
    def ask_chat_compact(chat_id: str, question: str = Form(...), generate_plot: bool = Form(default=True)) -> dict:
        """
        Ask a question with compact ChatResponse format (concise answer + key findings).
        
        Returns only the essential information:
        - question: The asked question
        - answer: Concise answer
        - key_findings: Top 3-5 findings
        - visual_insights: Chart/visual descriptions
        - plots_base64: Embedded visualization data
        - source_summary: Brief source description
        - data_types_used: Types of data referenced
        """
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        
        if not session.docs_dir.exists() or not any(path.is_file() for path in session.docs_dir.rglob("*")):
            logger.warning("Ask attempted before upload for chat %s", chat_id)
            raise HTTPException(status_code=400, detail="Upload a folder before asking questions")
        
        logger.info("Received compact question for chat %s: %s", chat_id, question)
        chat_settings = settings_factory.build(session)
        agent = LeadershipInsightAgent(chat_settings)
        response = agent.ask(question=question, docs_path=session.docs_dir, generate_plot=generate_plot)
        chat_manager.record_exchange(session, question, response)
        
        # Reload session to get updated title
        updated_session = chat_manager.load_chat(chat_id)
        logger.info("Answered question for chat %s", chat_id)
        
        # Convert to ChatResponse format
        chat_response = response.to_chat_response()
        response_dict = chat_response.to_dict()
        response_dict["title"] = updated_session.title  # Add the updated title
        
        return response_dict

    @app.get("/api/chats/{chat_id}/report/{question_slug}")
    def get_detailed_report(chat_id: str, question_slug: str) -> dict:
        """
        Get the detailed report for a specific question.
        
        Returns the full analysis including methodology, detailed findings,
        evidence bundles, and all source references.
        """
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        
        # Try to find the corresponding message and re-generate the report
        messages = chat_manager.load_messages(session)
        question = None
        
        # Find the question from messages
        for msg in messages:
            if msg.get("role") == "user":
                if slugify(msg.get("content", "")) == question_slug:
                    question = msg.get("content", "")
                    break
        
        if not question:
            raise HTTPException(status_code=404, detail="Question not found in chat history")
        
        logger.info("Generating detailed report for chat %s, question: %s", chat_id, question)
        chat_settings = settings_factory.build(session)
        agent = LeadershipInsightAgent(chat_settings)
        response = agent.ask(question=question, docs_path=session.docs_dir, generate_plot=False)
        
        # Convert to detailed report format
        detailed = response.to_detailed_report()
        
        # Format source references with URLs
        formatted_refs = _format_source_references(response.source_references, chat_id, session.docs_dir)
        
        return {
            "chat_id": chat_id,
            "question": detailed.question,
            "executive_summary": detailed.executive_summary,
            "planned_approach": detailed.planned_approach,
            "query_routing_logic": detailed.query_routing_logic,
            "detailed_findings": detailed.detailed_findings,
            "calculations": detailed.calculations,
            "caveats": detailed.caveats,
            "source_references": formatted_refs,
            "visual_insights": response.visual_insights,
        }

    @app.post("/api/chats/{chat_id}/clear")
    def clear_chat_analysis(chat_id: str) -> dict:
        """
        Clear all analysis data (plots, reports, conversation history) while keeping uploaded documents.
        This allows you to start fresh analysis without re-uploading files.
        """
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        chat_manager.clear_chat_data(session)
        logger.info("Cleared analysis data for chat %s", chat_id)
        return {"chat_id": chat_id, "cleared": True, "message": "Analysis data cleared. Uploaded documents are preserved."}

    @app.post("/api/chats/{chat_id}/delete-documents")
    def delete_uploaded_documents(chat_id: str) -> dict:
        """
        Delete all uploaded documents and dependent analysis data.
        This is useful when you want to remove documents without deleting the entire chat.
        The chat itself remains but will need new documents uploaded before analyzing.
        """
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        chat_manager.delete_uploaded_documents(session)
        logger.info("Deleted uploaded documents for chat %s", chat_id)
        return {
            "chat_id": chat_id,
            "deleted": True,
            "message": "All uploaded documents have been deleted. Chat remains available for new uploads.",
        }

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: str) -> dict:
        try:
            session = chat_manager.load_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Chat not found") from exc
        backend = build_memory_backend(settings_factory.build(session))
        if hasattr(backend, "delete_container"):
            backend.delete_container()
        chat_manager.delete_chat(session)
        logger.info("Deleted chat via API: %s", chat_id)
        return {"deleted": True}

    return app


def _format_source_references(references: list[str], chat_id: str, docs_dir: Path) -> list[str]:
    """Convert relative file paths in references to clickable artifact URLs."""
    import re
    
    formatted = []
    for ref in references:
        # Parse markdown link format: [display text](path)
        match = re.match(r'\[([^\]]+)\]\(([^\)]+)\)', ref)
        if match:
            display_text = match.group(1)
            file_path = match.group(2)
            
            # Try to resolve the file path
            try:
                resolved = docs_dir / file_path
                if resolved.exists() and resolved.is_file():
                    # Create artifact URL
                    artifact_url = f"/artifacts/{chat_id}/docs/{file_path}"
                    formatted.append(f"[{display_text}]({artifact_url})")
                else:
                    # Keep original if file not found
                    formatted.append(ref)
            except Exception:
                # Keep original if there's any error
                formatted.append(ref)
        else:
            # Not a markdown link, keep as is
            formatted.append(ref)
    
    return formatted


def _index_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Wald Decision Agent</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #f9f7f1;
      --panel-strong: #efe9de;
      --line: #d9d2c3;
      --ink: #1b1b1b;
      --muted: #5f5a52;
      --accent: #0f766e;
      --accent-2: #d97706;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family: "SF Pro Text", "Inter", ui-sans-serif, system-ui; background:radial-gradient(circle at top left, #fbfaf7, var(--bg)); color:var(--ink); }
    .app { display:grid; grid-template-columns: 300px 1fr; height:100vh; }
    .sidebar { border-right:1px solid var(--line); padding:18px; background:linear-gradient(180deg, #f1ebdf 0%, #ece5d8 100%); overflow:auto; }
    .main { display:flex; flex-direction:column; height:100vh; }
    .toolbar { padding:16px 20px; border-bottom:1px solid var(--line); background:rgba(255,255,255,0.72); backdrop-filter: blur(16px); display:flex; gap:12px; align-items:center; justify-content:space-between; }
    .messages { flex:1; overflow:auto; padding:24px; display:flex; flex-direction:column; gap:16px; background:linear-gradient(180deg, #fcfbf8 0%, #f7f4ee 100%); }
    .composer { padding:16px 20px; border-top:1px solid var(--line); background:#fbfaf7; display:flex; flex-direction:column; gap:12px; }
    .bubble { max-width:920px; padding:16px 18px; border-radius:18px; white-space:pre-wrap; line-height:1.45; box-shadow: 0 12px 28px rgba(27,27,27,0.06); }
    .user { align-self:flex-end; background:linear-gradient(135deg, #1b1b1b, #333); color:#fff; }
    .assistant { align-self:flex-start; background:#fff; border:1px solid #ddd4c7; }
    .chat-item { width:100%; text-align:left; padding:12px 14px; border-radius:14px; border:1px solid var(--line); background:#fff; margin-bottom:8px; cursor:pointer; }
    .chat-item.active { border-color:var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    textarea { width:100%; min-height:110px; border-radius:16px; border:1px solid #cfc4b2; padding:14px; resize:vertical; font:inherit; background:#fff; }
    button { border:none; border-radius:12px; padding:10px 14px; background:var(--ink); color:white; cursor:pointer; font-weight:600; }
    button:disabled { opacity:0.5; cursor:not-allowed; }
    .ghost { background:#fff; color:var(--ink); border:1px solid #cfc4b2; }
    .accent { background:linear-gradient(135deg, var(--accent), #155e75); }
    .warn { background:linear-gradient(135deg, var(--accent-2), #b45309); }
    .artifacts { margin:4px 0 8px 8px; }
    .artifacts a { display:inline-block; margin-right:10px; color:var(--accent); font-weight:600; }
    .hint { color:var(--muted); font-size:14px; }
    .panel { border:1px solid var(--line); background:rgba(255,255,255,0.82); border-radius:16px; padding:14px; margin-bottom:14px; }
    .stats { display:flex; gap:10px; flex-wrap:wrap; }
    .pill { padding:6px 10px; border-radius:999px; background:#fff; border:1px solid var(--line); font-size:13px; }
    .dropzone { position:relative; border:1.5px dashed #bdb29d; border-radius:16px; padding:14px; background:#fffdf8; }
    .dropzone input { width:100%; }
    .status-good { color:var(--accent); }
    .status-bad { color:#b91c1c; }
    .topline { display:flex; flex-direction:column; gap:2px; }
    .title { font-size:20px; font-weight:700; letter-spacing:-0.02em; }
    .subtitle { color:var(--muted); font-size:13px; }
    .markdown { white-space:pre-wrap; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { display:none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="topline" style="margin-bottom:16px;">
        <div class="title">Wald Decision Agent</div>
        <div class="subtitle">Chat-scoped document analysis</div>
      </div>
      <button class="accent" onclick="createChat()">New Chat</button>
      <div id="chatList" style="margin-top:16px;"></div>
    </aside>
    <main class="main">
      <div class="toolbar">
        <div class="topline">
          <div class="title" style="font-size:18px;">Document Chat</div>
          <div id="status" class="subtitle">Create a chat and upload a folder to begin.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
          <button class="ghost warn" onclick="deleteChat()">Delete Chat</button>
        </div>
      </div>
      <div id="messages" class="messages"></div>
      <div class="composer">
        <div class="panel">
          <div class="dropzone">
            <div style="font-weight:600; margin-bottom:6px;">Upload a folder</div>
            <div class="hint" style="margin-bottom:10px;">Selecting a folder starts ingestion immediately. If folder upload is unsupported in your browser, select multiple files instead.</div>
            <input id="folderInput" type="file" webkitdirectory directory multiple onchange="uploadFolder()" />
          </div>
          <div id="summary" class="stats" style="margin-top:12px;"></div>
        </div>
        <textarea id="question" placeholder="Ask a question about the uploaded documents..." disabled></textarea>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
          <span id="error" class="hint"></span>
          <button id="sendButton" onclick="askQuestion()" disabled>Send</button>
        </div>
      </div>
    </main>
  </div>
  <script>
    let activeChat = null;
    let activeDocCount = 0;

    function setStatus(message, tone='') {
      const status = document.getElementById('status');
      status.textContent = message;
      status.className = `subtitle ${tone}`.trim();
    }

    function setError(message='') {
      const error = document.getElementById('error');
      error.textContent = message;
      error.className = `hint ${message ? 'status-bad' : ''}`.trim();
    }

    function updateComposerState() {
      const enabled = activeChat && activeDocCount > 0;
      document.getElementById('question').disabled = !enabled;
      document.getElementById('sendButton').disabled = !enabled;
    }

    function renderSummary(chat) {
      const summary = document.getElementById('summary');
      summary.innerHTML = '';
      const pills = [
        `Chat ${chat.title}`,
        `Docs ${chat.doc_count}`,
        activeChat ? `DB ${chat.db_path.split('/').pop()}` : ''
      ].filter(Boolean);
      pills.forEach(text => {
        const pill = document.createElement('div');
        pill.className = 'pill';
        pill.textContent = text;
        summary.appendChild(pill);
      });
    }

    async function refreshChats() {
      const chats = await fetch('/api/chats').then(r => r.json());
      const list = document.getElementById('chatList');
      list.innerHTML = '';
      chats.forEach(chat => {
        const btn = document.createElement('button');
        btn.className = 'chat-item' + (chat.chat_id === activeChat ? ' active' : '');
        btn.textContent = chat.title;
        btn.onclick = () => loadChat(chat.chat_id);
        list.appendChild(btn);
      });
      if (!activeChat && chats.length) {
        await loadChat(chats[0].chat_id);
      }
    }

    async function createChat() {
      const form = new FormData();
      const chat = await fetch('/api/chats', { method:'POST', body: form }).then(r => r.json());
      activeChat = chat.chat_id;
      activeDocCount = 0;
      setError('');
      setStatus('Chat created. Upload a folder to start ingestion.');
      await refreshChats();
      await loadChat(chat.chat_id);
    }

    async function loadChat(chatId) {
      activeChat = chatId;
      const chat = await fetch(`/api/chats/${chatId}`).then(r => r.json());
      const messages = document.getElementById('messages');
      messages.innerHTML = '';
      chat.messages.forEach(message => renderMessage(message));
      activeDocCount = chat.doc_count;
      renderSummary(chat);
      setStatus(chat.doc_count ? `Ready. ${chat.doc_count} document files available.` : 'Upload a folder to start ingestion.', chat.doc_count ? 'status-good' : '');
      updateComposerState();
      await refreshChats();
    }

    function renderMessage(message) {
      const messages = document.getElementById('messages');
      const bubble = document.createElement('div');
      bubble.className = 'bubble ' + (message.role === 'user' ? 'user' : 'assistant');
      bubble.textContent = message.role === 'assistant' ? message.markdown || message.content : message.content;
      messages.appendChild(bubble);
      if (message.role === 'assistant' && (message.plot_paths || message.report_path)) {
        const artifacts = document.createElement('div');
        artifacts.className = 'artifacts';
        if (message.plot_paths) {
          message.plot_paths.forEach(path => {
            const a = document.createElement('a');
            a.href = `/artifacts/${activeChat}/artifacts/plots/${path.split('/').pop()}`;
            a.textContent = 'Plot';
            a.target = '_blank';
            artifacts.appendChild(a);
          });
        }
        if (message.report_path) {
          const a = document.createElement('a');
          a.href = `/artifacts/${activeChat}/artifacts/reports/${message.report_path.split('/').pop()}`;
          a.textContent = 'Report';
          a.target = '_blank';
          artifacts.appendChild(a);
        }
        messages.appendChild(artifacts);
      }
      messages.scrollTop = messages.scrollHeight;
    }

    async function uploadFolder() {
      if (!activeChat) { await createChat(); }
      const input = document.getElementById('folderInput');
      if (!input.files.length) {
        setError('Select a folder or multiple files first.');
        return;
      }
      setError('');
      setStatus(`Uploading ${input.files.length} files and starting ingestion...`);
      const form = new FormData();
      Array.from(input.files).forEach(file => {
        form.append('files', file, file.name);
        form.append('relative_paths', file.webkitRelativePath || file.name);
      });
      const response = await fetch(`/api/chats/${activeChat}/upload`, { method:'POST', body: form });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || 'Upload failed.');
        setStatus('Upload failed.', 'status-bad');
        return;
      }
      activeDocCount = data.documents;
      setStatus(`Uploaded ${data.file_count} files. Ingested ${data.documents} documents, ${data.tables} tables, ${data.visuals} visuals.`, 'status-good');
      updateComposerState();
      await loadChat(activeChat);
    }

    async function askQuestion() {
      if (!activeChat) { await createChat(); }
      const question = document.getElementById('question').value.trim();
      if (!question) { return; }
      if (!activeDocCount) {
        setError('Upload a folder before asking questions.');
        return;
      }
      setError('');
      setStatus('Running retrieval and analysis...');
      renderMessage({ role:'user', content: question });
      document.getElementById('question').value = '';
      const form = new FormData();
      form.append('question', question);
      form.append('generate_plot', 'true');
      const raw = await fetch(`/api/chats/${activeChat}/ask`, { method:'POST', body: form });
      const response = await raw.json();
      if (!raw.ok) {
        setError(response.detail || 'Question failed.');
        setStatus('Question failed.', 'status-bad');
        await loadChat(activeChat);
        return;
      }
      renderMessage({ role:'assistant', markdown: response.markdown, plot_paths: response.plot_urls, report_path: response.report_url });
      setStatus('Answer generated.', 'status-good');
      await loadChat(activeChat);
    }

    async function deleteChat() {
      if (!activeChat) { return; }
      await fetch(`/api/chats/${activeChat}`, { method:'DELETE' });
      activeChat = null;
      activeDocCount = 0;
      document.getElementById('messages').innerHTML = '';
      document.getElementById('summary').innerHTML = '';
      setError('');
      setStatus('Chat deleted.');
      updateComposerState();
      await refreshChats();
    }

    refreshChats();
  </script>
</body>
</html>
"""
