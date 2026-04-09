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
from ..reasoning.conversation import ConversationContextResolver
from ..utils import slugify


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    configure_logging(app_settings)
    logger = get_logger("web.app")
    chat_manager = ChatManager(app_settings)
    settings_factory = ChatSettingsFactory(app_settings)
    context_resolver = ConversationContextResolver()

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
        resolved = context_resolver.resolve(question, chat_manager.load_messages(session))
        if resolved.used_history:
            logger.info("Resolved follow-up question for chat %s to: %s", chat_id, resolved.question)
        response = agent.ask(question=resolved.question, docs_path=session.docs_dir, generate_plot=generate_plot)
        
        # Reload session to get updated title
        chat_response = response.to_chat_response()
        plot_urls = [f"/artifacts/{chat_id}/artifacts/plots/{Path(path).name}" for path in response.plot_paths]
        formatted_evidence = _format_inline_references(chat_response.evidence, chat_id, session.docs_dir)
        chat_markdown = _build_chat_markdown(
            answer=chat_response.answer,
            key_findings=chat_response.key_findings,
            evidence=formatted_evidence,
            plot_urls=plot_urls,
        )
        chat_manager.record_exchange(
            session,
            question,
            response,
            evidence=formatted_evidence,
            plot_urls=plot_urls,
            markdown=chat_markdown,
        )

        updated_session = chat_manager.load_chat(chat_id)
        logger.info("Answered question for chat %s", chat_id)
        
        return {
            "chat_id": chat_id,
            "title": updated_session.title,  # DYNAMICALLY UPDATED TITLE
            "question": question,
            "answer": chat_response.answer,
            "key_findings": chat_response.key_findings,
            "evidence": formatted_evidence,
            "visual_insights": chat_response.visual_insights,
            "plot_urls": plot_urls,
            "plot_base64": chat_response.plots_base64[0] if chat_response.plots_base64 else "",
            "source_summary": chat_response.source_summary,
            "data_types_used": chat_response.data_types_used,
            "report_url": f"/artifacts/{chat_id}/artifacts/reports/{slugify(question)}.md",
            "markdown": chat_markdown,
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
        resolved = context_resolver.resolve(question, chat_manager.load_messages(session))
        if resolved.used_history:
            logger.info("Resolved follow-up question for chat %s to: %s", chat_id, resolved.question)
        response = agent.ask(question=resolved.question, docs_path=session.docs_dir, generate_plot=generate_plot)
        plot_urls = [f"/artifacts/{chat_id}/artifacts/plots/{Path(path).name}" for path in response.plot_paths]
        chat_response = response.to_chat_response()
        formatted_evidence = _format_inline_references(chat_response.evidence, chat_id, session.docs_dir)
        chat_markdown = _build_chat_markdown(
            answer=chat_response.answer,
            key_findings=chat_response.key_findings,
            evidence=formatted_evidence,
            plot_urls=plot_urls,
        )
        chat_manager.record_exchange(
            session,
            question,
            response,
            evidence=formatted_evidence,
            plot_urls=plot_urls,
            markdown=chat_markdown,
        )
        
        # Reload session to get updated title
        updated_session = chat_manager.load_chat(chat_id)
        logger.info("Answered question for chat %s", chat_id)
        
        # Convert to ChatResponse format
        response_dict = chat_response.to_dict()
        response_dict["title"] = updated_session.title  # Add the updated title
        response_dict["evidence"] = formatted_evidence
        response_dict["plot_urls"] = plot_urls
        response_dict["markdown"] = chat_markdown
        
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
        match = re.match(r'\[(.+)\]\(([^)]+)\)', ref)
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


def _format_inline_references(items: list[str], chat_id: str, docs_dir: Path) -> list[str]:
    """Rewrite markdown links inside evidence lines to chat-scoped artifact URLs."""
    import re

    pattern = re.compile(r"\[(.+)\]\(([^)]+)\)")
    formatted_items: list[str] = []
    for item in items:
        def replace(match: re.Match[str]) -> str:
            label = match.group(1)
            file_path = match.group(2)
            resolved = docs_dir / file_path
            if resolved.exists() and resolved.is_file():
                return f"[{label}](/artifacts/{chat_id}/docs/{file_path})"
            return match.group(0)

        formatted_items.append(pattern.sub(replace, item))
    return formatted_items


def _build_chat_markdown(answer: str, key_findings: list[str], evidence: list[str], plot_urls: list[str]) -> str:
    sections = [
        ("Executive Summary", answer),
        ("Key Findings", "\n".join(f"{idx}. {item}" for idx, item in enumerate(key_findings, start=1)) or "1. No findings generated."),
        ("Evidence", "\n".join(f"- {item}" for item in evidence) or "- No evidence retrieved."),
    ]
    if plot_urls:
        sections.append(("Plots", "\n".join(f"![plot]({url})" for url in plot_urls)))
    return "\n\n".join(f"{title}\n{body}" for title, body in sections)


def _index_html() -> str:
    return r"""
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
    .bubble { max-width:920px; padding:16px 18px; border-radius:18px; line-height:1.45; box-shadow: 0 12px 28px rgba(27,27,27,0.06); }
    .user { align-self:flex-end; background:linear-gradient(135deg, #1b1b1b, #333); color:#fff; }
    .assistant { align-self:flex-start; background:#fff; border:1px solid #ddd4c7; }
    .chat-item { width:100%; text-align:left; padding:0; border-radius:16px; border:1px solid var(--line); background:#fff; margin-bottom:10px; cursor:pointer; overflow:hidden; }
    .chat-item.active { border-color:var(--accent); box-shadow: inset 0 0 0 1px var(--accent), 0 10px 24px rgba(15,118,110,0.08); }
    .chat-card { padding:12px 14px; display:flex; flex-direction:column; gap:6px; }
    .chat-card-title { font-weight:700; color:var(--ink); line-height:1.35; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .chat-card-meta { display:flex; gap:8px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
    .chat-empty { color:var(--muted); font-size:13px; padding:10px 4px; }
    textarea { width:100%; min-height:110px; border-radius:16px; border:1px solid #cfc4b2; padding:14px; resize:vertical; font:inherit; background:#fff; }
    button { border:none; border-radius:12px; padding:10px 14px; background:var(--ink); color:white; cursor:pointer; font-weight:600; }
    button:disabled { opacity:0.5; cursor:not-allowed; }
    .ghost { background:#fff; color:var(--ink); border:1px solid #cfc4b2; }
    .accent { background:linear-gradient(135deg, var(--accent), #155e75); }
    .warn { background:linear-gradient(135deg, var(--accent-2), #b45309); }
    .artifacts { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
    .artifacts a { display:inline-block; color:var(--accent); font-weight:600; text-decoration:none; }
    .hint { color:var(--muted); font-size:14px; }
    .panel { border:1px solid var(--line); background:rgba(255,255,255,0.82); border-radius:16px; padding:14px; margin-bottom:14px; }
    .stats { display:flex; gap:10px; flex-wrap:wrap; }
    .pill { padding:6px 10px; border-radius:999px; background:#fff; border:1px solid var(--line); font-size:13px; }
    .dropzone { position:relative; border:1.5px dashed #bdb29d; border-radius:16px; padding:14px; background:#fffdf8; display:flex; flex-direction:column; gap:12px; }
    .dropzone input { display:none; }
    .upload-actions { display:flex; gap:10px; flex-wrap:wrap; }
    .upload-actions button { background:#fff; color:var(--ink); border:1px solid #cfc4b2; }
    .upload-actions button.primary-action { background:linear-gradient(135deg, var(--accent), #155e75); color:#fff; border:none; }
    .upload-actions button.warn-action { background:linear-gradient(135deg, var(--accent-2), #b45309); color:#fff; border:none; }
    .upload-note { color:var(--muted); font-size:13px; }
    .doc-panel { display:flex; flex-direction:column; gap:10px; }
    .doc-panel-header { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .doc-panel-title { display:flex; flex-direction:column; gap:2px; min-width:0; }
    .doc-panel-title strong { font-size:14px; }
    .doc-panel-toggle { background:#fff; color:var(--ink); border:1px solid #cfc4b2; padding:8px 12px; }
    .doc-panel-body[hidden] { display:none; }
    .doc-summary-inline { color:var(--muted); font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .status-good { color:var(--accent); }
    .status-bad { color:#b91c1c; }
    .topline { display:flex; flex-direction:column; gap:2px; }
    .title { font-size:20px; font-weight:700; letter-spacing:-0.02em; }
    #chatHeading { max-width:620px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .subtitle { color:var(--muted); font-size:13px; }
    .assistant-body { display:flex; flex-direction:column; gap:14px; }
    .assistant-section { display:flex; flex-direction:column; gap:8px; }
    .assistant-section h4 { margin:0; font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--muted); }
    .assistant-section p { margin:0; }
    .assistant-section ol, .assistant-section ul { margin:0; padding-left:20px; }
    .assistant-section li { margin:0 0 6px; }
    .assistant-section a { color:var(--accent); }
    .plot-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px; }
    .plot-card { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fffdf8; }
    .plot-card img { display:block; width:100%; height:auto; background:#fff; }
    .plot-card a { display:block; padding:10px 12px; color:var(--accent); font-weight:600; text-decoration:none; border-top:1px solid var(--line); }
    .plot-carousel { display:flex; flex-direction:column; gap:10px; }
    .plot-carousel-frame { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fffdf8; }
    .plot-carousel-frame img { display:block; width:100%; height:auto; background:#fff; }
    .plot-carousel-controls { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .plot-nav { min-width:42px; padding:8px 12px; border-radius:999px; background:#fff; color:var(--ink); border:1px solid var(--line); font-size:18px; line-height:1; }
    .plot-nav:disabled { opacity:0.4; cursor:not-allowed; }
    .plot-meta { display:flex; align-items:center; gap:10px; flex:1; justify-content:flex-end; }
    .plot-counter { color:var(--muted); font-size:13px; min-width:52px; text-align:center; }
    .plot-open-link { color:var(--accent); font-weight:600; text-decoration:none; }
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
          <div id="chatHeading" class="title" style="font-size:18px;">Document Chat</div>
          <div id="status" class="subtitle">Create a chat and upload a folder to begin.</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
          <button class="ghost warn" onclick="deleteChat()">Delete Chat</button>
        </div>
      </div>
      <div id="messages" class="messages"></div>
      <div class="composer">
        <div class="panel">
          <div class="doc-panel">
            <div class="doc-panel-header">
              <div class="doc-panel-title">
                <strong>Documents</strong>
                <div id="docSummaryInline" class="doc-summary-inline">Create or open a chat, then choose a document action.</div>
              </div>
              <button id="docPanelToggle" type="button" class="doc-panel-toggle" onclick="toggleDocumentPanel()">Manage</button>
            </div>
            <div id="docPanelBody" class="doc-panel-body">
              <div class="dropzone">
                <div class="hint">Add more files to the current chat, replace the current document set, or remove uploaded files entirely.</div>
                <div class="upload-actions">
                  <button type="button" class="primary-action" onclick="openUploadPicker('add')">Add Documents</button>
                  <button type="button" onclick="openUploadPicker('replace')">Replace Documents</button>
                  <button type="button" class="warn-action" onclick="deleteDocuments()">Delete Documents</button>
                </div>
                <div id="uploadModeLabel" class="upload-note">Select a chat, then choose a document action.</div>
                <input id="folderInput" type="file" webkitdirectory directory multiple onchange="uploadFolder()" />
              </div>
            </div>
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
    let pendingUploadMode = 'add';
    let documentPanelExpanded = true;

    function displayTitle(chat) {
      const raw = (chat && chat.title ? chat.title : '').trim();
      return raw || 'New Chat';
    }

    function formatTimestamp(value) {
      if (!value) return '';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return '';
      return date.toLocaleString([], { month:'short', day:'numeric', hour:'numeric', minute:'2-digit' });
    }

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
      const uploadLabel = document.getElementById('uploadModeLabel');
      const inlineSummary = document.getElementById('docSummaryInline');
      if (!activeChat) {
        uploadLabel.textContent = 'Create or open a chat, then choose a document action.';
        inlineSummary.textContent = 'Create or open a chat, then choose a document action.';
        setDocumentPanelExpanded(true);
        return;
      }
      if (!activeDocCount) {
        uploadLabel.textContent = 'No documents uploaded yet. Start by adding a folder or selecting multiple files.';
        inlineSummary.textContent = 'No documents uploaded yet.';
        setDocumentPanelExpanded(true);
        return;
      }
      uploadLabel.textContent = `${activeDocCount} files are available in this chat. Add more documents, replace them, or delete them.`;
      inlineSummary.textContent = `${activeDocCount} files available for this chat.`;
      if (pendingUploadMode === 'add') {
        setDocumentPanelExpanded(false);
      }
    }

    function setDocumentPanelExpanded(expanded) {
      documentPanelExpanded = expanded;
      const body = document.getElementById('docPanelBody');
      const toggle = document.getElementById('docPanelToggle');
      body.hidden = !expanded;
      toggle.textContent = expanded ? 'Hide' : 'Manage';
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }

    function toggleDocumentPanel(forceState = null) {
      const nextState = forceState === null ? !documentPanelExpanded : forceState;
      setDocumentPanelExpanded(nextState);
    }

    function renderSummary(chat) {
      const summary = document.getElementById('summary');
      summary.innerHTML = '';
      const pills = [
        `Chat ${displayTitle(chat)}`,
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

    function renderChatHeading(chat) {
      const heading = document.getElementById('chatHeading');
      const text = chat ? displayTitle(chat) : 'Document Chat';
      heading.textContent = text;
      heading.title = text;
    }

    async function refreshChats() {
      const chats = await fetch('/api/chats').then(r => r.json());
      const list = document.getElementById('chatList');
      list.innerHTML = '';
      if (!chats.length) {
        const empty = document.createElement('div');
        empty.className = 'chat-empty';
        empty.textContent = 'No chats yet.';
        list.appendChild(empty);
      }
      chats.forEach(chat => {
        const btn = document.createElement('button');
        btn.className = 'chat-item' + (chat.chat_id === activeChat ? ' active' : '');
        const card = document.createElement('div');
        card.className = 'chat-card';
        const title = document.createElement('div');
        title.className = 'chat-card-title';
        title.textContent = displayTitle(chat);
        title.title = displayTitle(chat);
        const meta = document.createElement('div');
        meta.className = 'chat-card-meta';
        const metaParts = [
          chat.doc_count ? `${chat.doc_count} docs` : 'No docs',
          formatTimestamp(chat.updated_at)
        ].filter(Boolean);
        meta.textContent = metaParts.join(' • ');
        card.appendChild(title);
        card.appendChild(meta);
        btn.appendChild(card);
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
      pendingUploadMode = 'add';
      setDocumentPanelExpanded(true);
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
      renderChatHeading(chat);
      renderSummary(chat);
      setStatus(chat.doc_count ? `Ready. ${chat.doc_count} document files available.` : 'Upload a folder to start ingestion.', chat.doc_count ? 'status-good' : '');
      setDocumentPanelExpanded(chat.doc_count === 0);
      updateComposerState();
      await refreshChats();
    }

    function openUploadPicker(mode) {
      pendingUploadMode = mode;
      const input = document.getElementById('folderInput');
      const modeLabel = document.getElementById('uploadModeLabel');
      if (mode === 'replace') {
        modeLabel.textContent = 'Replace mode: selecting files will remove the current document set and ingest the new one.';
      } else {
        modeLabel.textContent = 'Add mode: selecting files will keep the current documents and ingest the new ones too.';
      }
      setDocumentPanelExpanded(true);
      input.value = '';
      input.click();
    }

    function escapeHtml(value) {
      return (value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderInlineMarkdown(text) {
      const escaped = escapeHtml(text);
      return escaped.replace(/\[(.+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    function appendAssistantSection(container, title, items, ordered=false) {
      if (!items || (Array.isArray(items) && !items.length)) {
        return;
      }
      const section = document.createElement('section');
      section.className = 'assistant-section';
      const heading = document.createElement('h4');
      heading.textContent = title;
      section.appendChild(heading);

      if (Array.isArray(items)) {
        const list = document.createElement(ordered ? 'ol' : 'ul');
        items.forEach(item => {
          const li = document.createElement('li');
          li.innerHTML = renderInlineMarkdown(item);
          list.appendChild(li);
        });
        section.appendChild(list);
      } else {
        const paragraph = document.createElement('p');
        paragraph.innerHTML = renderInlineMarkdown(items);
        section.appendChild(paragraph);
      }

      container.appendChild(section);
    }

    function createPlotCarousel(plotUrls) {
      let activeIndex = 0;
      const wrapper = document.createElement('div');
      wrapper.className = 'plot-carousel';

      const frame = document.createElement('div');
      frame.className = 'plot-carousel-frame';
      const img = document.createElement('img');
      frame.appendChild(img);

      const controls = document.createElement('div');
      controls.className = 'plot-carousel-controls';

      const prev = document.createElement('button');
      prev.className = 'plot-nav';
      prev.textContent = '←';
      prev.type = 'button';

      const next = document.createElement('button');
      next.className = 'plot-nav';
      next.textContent = '→';
      next.type = 'button';

      const meta = document.createElement('div');
      meta.className = 'plot-meta';
      const counter = document.createElement('div');
      counter.className = 'plot-counter';
      const link = document.createElement('a');
      link.className = 'plot-open-link';
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'Open plot';
      meta.appendChild(counter);
      meta.appendChild(link);

      function syncPlot() {
        const url = plotUrls[activeIndex];
        img.src = url;
        img.alt = `Plot ${activeIndex + 1}`;
        link.href = url;
        counter.textContent = `${activeIndex + 1} / ${plotUrls.length}`;
        prev.disabled = activeIndex === 0;
        next.disabled = activeIndex === plotUrls.length - 1;
      }

      prev.onclick = () => {
        if (activeIndex > 0) {
          activeIndex -= 1;
          syncPlot();
        }
      };

      next.onclick = () => {
        if (activeIndex < plotUrls.length - 1) {
          activeIndex += 1;
          syncPlot();
        }
      };

      controls.appendChild(prev);
      controls.appendChild(meta);
      controls.appendChild(next);

      wrapper.appendChild(frame);
      wrapper.appendChild(controls);
      syncPlot();
      return wrapper;
    }

    function renderAssistantMessage(message, bubble) {
      const body = document.createElement('div');
      body.className = 'assistant-body';
      appendAssistantSection(body, 'Executive Summary', message.answer || message.content || '');
      appendAssistantSection(body, 'Key Findings', message.key_findings || [], true);
      appendAssistantSection(body, 'Evidence', message.evidence || []);

      const plotUrls = message.plot_urls || [];
      const hasEmbeddedPlot = message.plot_base64 && message.plot_base64.length > 0;
      if (plotUrls.length || hasEmbeddedPlot) {
        const section = document.createElement('section');
        section.className = 'assistant-section';
        const heading = document.createElement('h4');
        heading.textContent = 'Plots';
        section.appendChild(heading);

        if (plotUrls.length > 1) {
          section.appendChild(createPlotCarousel(plotUrls));
        } else if (plotUrls.length === 1) {
          const grid = document.createElement('div');
          grid.className = 'plot-grid';
          const card = document.createElement('div');
          card.className = 'plot-card';
          const img = document.createElement('img');
          img.src = plotUrls[0];
          img.alt = 'Plot 1';
          card.appendChild(img);
          const link = document.createElement('a');
          link.href = plotUrls[0];
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = 'Open plot';
          card.appendChild(link);
          grid.appendChild(card);
          section.appendChild(grid);
        } else if (hasEmbeddedPlot) {
          const card = document.createElement('div');
          card.className = 'plot-card';
          const img = document.createElement('img');
          img.src = `data:image/png;base64,${message.plot_base64}`;
          img.alt = 'Generated plot';
          card.appendChild(img);
          const grid = document.createElement('div');
          grid.className = 'plot-grid';
          grid.appendChild(card);
          section.appendChild(grid);
        }
        body.appendChild(section);
      }

      if (message.report_url || message.report_path) {
        const artifacts = document.createElement('div');
        artifacts.className = 'artifacts';
        const reportLink = document.createElement('a');
        reportLink.href = message.report_url || `/artifacts/${activeChat}/artifacts/reports/${message.report_path.split('/').pop()}`;
        reportLink.textContent = 'Open detailed report';
        reportLink.target = '_blank';
        reportLink.rel = 'noopener noreferrer';
        artifacts.appendChild(reportLink);
        body.appendChild(artifacts);
      }

      bubble.appendChild(body);
    }

    function renderMessage(message) {
      const messages = document.getElementById('messages');
      const bubble = document.createElement('div');
      bubble.className = 'bubble ' + (message.role === 'user' ? 'user' : 'assistant');
      if (message.role === 'assistant') {
        renderAssistantMessage(message, bubble);
      } else {
        bubble.textContent = message.content;
      }
      messages.appendChild(bubble);
      messages.scrollTop = messages.scrollHeight;
    }

    async function uploadFolder() {
      if (!activeChat) { await createChat(); }
      const input = document.getElementById('folderInput');
      if (!input.files.length) {
        setError('Select a folder or multiple files first.');
        return;
      }
      const incremental = pendingUploadMode !== 'replace';
      if (!incremental) {
        const confirmed = window.confirm('Replace the current document set for this chat? Existing uploaded files and derived analysis will be removed.');
        if (!confirmed) {
          input.value = '';
          updateComposerState();
          return;
        }
      }
      setError('');
      setStatus(`${incremental ? 'Adding' : 'Replacing with'} ${input.files.length} files and starting ingestion...`);
      const form = new FormData();
      Array.from(input.files).forEach(file => {
        form.append('files', file, file.name);
        form.append('relative_paths', file.webkitRelativePath || file.name);
      });
      form.append('incremental', incremental ? 'true' : 'false');
      const response = await fetch(`/api/chats/${activeChat}/upload`, { method:'POST', body: form });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || 'Upload failed.');
        setStatus('Upload failed.', 'status-bad');
        input.value = '';
        return;
      }
      activeDocCount = data.documents;
      setStatus(`${incremental ? 'Added' : 'Replaced with'} ${data.file_count} files. Ingested ${data.documents} documents, ${data.tables} tables, ${data.visuals} visuals.`, 'status-good');
      input.value = '';
      pendingUploadMode = 'add';
      setDocumentPanelExpanded(false);
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
      renderMessage({
        role:'assistant',
        content: response.answer,
        answer: response.answer,
        key_findings: response.key_findings,
        evidence: response.evidence,
        plot_urls: response.plot_urls,
        plot_base64: response.plot_base64,
        report_url: response.report_url,
        markdown: response.markdown
      });
      setStatus('Answer generated.', 'status-good');
      await loadChat(activeChat);
    }

    async function deleteChat() {
      if (!activeChat) { return; }
      await fetch(`/api/chats/${activeChat}`, { method:'DELETE' });
      activeChat = null;
      activeDocCount = 0;
      pendingUploadMode = 'add';
      document.getElementById('messages').innerHTML = '';
      document.getElementById('summary').innerHTML = '';
      renderChatHeading(null);
      setError('');
      setStatus('Chat deleted.');
      setDocumentPanelExpanded(true);
      updateComposerState();
      await refreshChats();
    }

    async function deleteDocuments() {
      if (!activeChat) { return; }
      if (!activeDocCount) {
        setError('There are no uploaded documents to delete for this chat.');
        return;
      }
      const confirmed = window.confirm('Delete all uploaded documents for this chat? Existing analysis will also be cleared.');
      if (!confirmed) {
        return;
      }
      setError('');
      setStatus('Deleting uploaded documents...');
      const raw = await fetch(`/api/chats/${activeChat}/delete-documents`, { method:'POST' });
      const response = await raw.json();
      if (!raw.ok) {
        setError(response.detail || 'Failed to delete uploaded documents.');
        setStatus('Document deletion failed.', 'status-bad');
        return;
      }
      activeDocCount = 0;
      pendingUploadMode = 'add';
      document.getElementById('messages').innerHTML = '';
      document.getElementById('question').value = '';
      setStatus('Uploaded documents deleted. Add a new folder to continue.', 'status-good');
      setDocumentPanelExpanded(true);
      updateComposerState();
      await loadChat(activeChat);
    }

    refreshChats();
  </script>
</body>
</html>
"""
