"""
kb-sorter-pipeline — MISTR Knowledge Base Router + Sorter Agents
POST /sort   → routes Oracle Export markdown into 12 KB domains and writes .md files to GitHub
GET  /kb-status → returns file counts per KB repo
GET  /health → liveness check
"""

import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from typing import Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GITHUB_TOKEN      = os.environ["GITHUB_TOKEN"]
GITHUB_ORG        = os.environ.get("GITHUB_ORG", "Ad-ROI")
COMMITTER_NAME    = os.environ.get("COMMITTER_NAME", "MISTR KB Sorter")
COMMITTER_EMAIL   = os.environ.get("COMMITTER_EMAIL", "mistr@mindstream.ing")
ROUTER_MODEL      = "claude-haiku-4-5"

# ─── 12 KB Domain Definitions ──────────────────────────────────────────────────

KB_DOMAINS = [
    {
        "id": "brand-persona",
        "repo": "kb-brand-persona",
        "description": (
            "Brand identity, mission, values, voice, tone, messaging, taglines, analogies, "
            "emotional storytelling, David's personal story, company positioning, marketing copy, "
            "MindStream.ing brand rules, MISTR name origin"
        ),
    },
    {
        "id": "creative-design",
        "repo": "kb-creative-design",
        "description": (
            "Visual design, UI/UX, creative assets, mockups, design systems, color palettes, "
            "typography, logos, imagery, video production, Remotion, visual standards"
        ),
    },
    {
        "id": "client-management",
        "repo": "kb-client-management",
        "description": (
            "Client relationships, onboarding workflows, account management, client communication, "
            "deliverables, contracts, client-facing SOPs, client portals"
        ),
    },
    {
        "id": "staff-management",
        "repo": "kb-staff-management",
        "description": (
            "Team management, hiring, HR, contractors, offboarding, performance reviews, "
            "team communication, roles, responsibilities, Desklog, former employees"
        ),
    },
    {
        "id": "ai-agents-registry",
        "repo": "kb-ai-agents-registry",
        "description": (
            "AI agent definitions, capabilities, skills, agent hierarchy, MISTR, PAM, sorter agents, "
            "MCP servers, agent builds, Claude API, Anthropic SDK, agent configurations, tool specs"
        ),
    },
    {
        "id": "dri-integrations",
        "repo": "kb-dri-integrations",
        "description": (
            "Third-party integrations, APIs, webhooks, Zapier, Activepieces, connected services, "
            "DRI ownership, authentication, Twilio, Slack, GitHub, GCP, OAuth, MCP connections"
        ),
    },
    {
        "id": "analytics-reporting",
        "repo": "kb-analytics-reporting",
        "description": (
            "Analytics, reporting, metrics, KPIs, dashboards, data tracking, performance measurement, "
            "business intelligence, conversion rates, revenue tracking"
        ),
    },
    {
        "id": "crm-automations",
        "repo": "kb-crm-automations",
        "description": (
            "CRM workflows, 1Dash.Pro automations, email sequences, pipelines, lead capture, "
            "follow-up sequences, funnels, sales automation, GoHighLevel, campaign management"
        ),
    },
    {
        "id": "project-mgmt-sops",
        "repo": "kb-project-mgmt-sops",
        "description": (
            "Project management, SOPs, standard operating procedures, operational workflows, "
            "Dart tasks, recurring processes, checklists, session protocols, rules of engagement"
        ),
    },
    {
        "id": "sales-management",
        "repo": "kb-sales-management",
        "description": (
            "Sales strategy, objection handling, pricing, proposals, closing techniques, "
            "sales psychology, client acquisition, revenue, pitch structure, offer design"
        ),
    },
    {
        "id": "personal",
        "repo": "kb-personal",
        "description": (
            "Personal notes, David's personal context, family, health, personal goals, "
            "mindset, non-business topics, private information"
        ),
    },
    {
        "id": "platform-technical",
        "repo": "kb-platform-technical",
        "description": (
            "Technical infrastructure, GCP architecture, Cloud Run services, database schemas, "
            "system design, code, devops, deployment configs, Docker, env vars, security, billing"
        ),
    },
]

DOMAIN_MAP = {d["id"]: d for d in KB_DOMAINS}

# ─── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="KB Sorter Pipeline", version="1.0.0")
_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Pydantic Models ────────────────────────────────────────────────────────────

class SortRequest(BaseModel):
    content: str
    session_date: Optional[str] = None

class DomainResult(BaseModel):
    domain: str
    repo: str
    status: str
    path: Optional[str] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None

class SortResponse(BaseModel):
    status: str
    session_date: str
    files_written: int
    domains_skipped: int
    domains_errored: int
    results: list[DomainResult]

# ─── GitHub Helpers ─────────────────────────────────────────────────────────────

GH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


async def gh_get_file(client: httpx.AsyncClient, repo: str, path: str) -> tuple[Optional[str], Optional[str]]:
    """Return (decoded_content, sha) or (None, None) if file not found."""
    url = f"https://api.github.com/repos/{GITHUB_ORG}/{repo}/contents/{path}"
    r = await client.get(url, headers=GH_HEADERS)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"].replace("\n", "")).decode("utf-8")
    return content, data["sha"]


async def gh_write_file(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    content: str,
    message: str,
    sha: Optional[str] = None,
) -> dict:
    """Create or update a file. Returns GitHub API response."""
    url = f"https://api.github.com/repos/{GITHUB_ORG}/{repo}/contents/{path}"
    payload: dict = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "committer": {"name": COMMITTER_NAME, "email": COMMITTER_EMAIL},
    }
    if sha:
        payload["sha"] = sha
    r = await client.put(url, headers=GH_HEADERS, json=payload)
    r.raise_for_status()
    return r.json()


async def gh_list_md_files(client: httpx.AsyncClient, repo: str, path: str = "") -> list[dict]:
    """List .md files at a given path in the repo (non-recursive, top level only)."""
    url = f"https://api.github.com/repos/{GITHUB_ORG}/{repo}/contents/{path}"
    r = await client.get(url, headers=GH_HEADERS)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        return []
    return [f for f in items if f.get("type") == "file" and f.get("name", "").endswith(".md")]


async def gh_count_all_md(client: httpx.AsyncClient, repo: str) -> int:
    """Count total .md files at root of repo."""
    files = await gh_list_md_files(client, repo)
    return len(files)

# ─── Router (claude-haiku-4-5) ──────────────────────────────────────────────────

def build_router_prompt(oracle_export: str) -> str:
    domains_json = json.dumps(
        [{"id": d["id"], "description": d["description"]} for d in KB_DOMAINS],
        indent=2,
    )
    return f"""You are the MISTR Knowledge Base Router for MindStream.ing.

You receive an Oracle Export document (structured session notes from an AI business operating system).
Your job: parse the export and split its content into 12 domain-specific knowledge base chunks.

THE 12 KB DOMAINS:
{domains_json}

ROUTING RULES:
- Route each piece of information to ALL domains it belongs to (content can be in multiple domains).
- Brand stories, analogies, voice rules → brand-persona
- Visual/design/creative work → creative-design
- Client relationships, onboarding, deliverables → client-management
- Team, HR, staff, contractors, offboarding → staff-management
- AI agents, MCP servers, MISTR capabilities, Claude API → ai-agents-registry
- APIs, webhooks, third-party integrations, DRI ownership → dri-integrations
- Metrics, KPIs, reporting, dashboards → analytics-reporting
- 1Dash.Pro, CRM, email sequences, funnels, automations → crm-automations
- SOPs, processes, Dart tasks, checklists, operational rules → project-mgmt-sops
- Sales strategy, pricing, objections, closing, offers → sales-management
- Personal/family/health/private content → personal
- GCP, Cloud Run, code, devops, infrastructure, Docker → platform-technical
- If a domain has NO relevant content, return an empty string for it.

ORACLE EXPORT:
{oracle_export}

Return ONLY a valid JSON object with exactly these 12 keys. Each value is a markdown string.
Keys (exact): brand-persona, creative-design, client-management, staff-management,
ai-agents-registry, dri-integrations, analytics-reporting, crm-automations,
project-mgmt-sops, sales-management, personal, platform-technical

No extra text, no markdown fences, just the JSON object."""


def route_content(oracle_export: str) -> dict[str, str]:
    """Call claude-haiku-4-5 to split content into 12 domain chunks."""
    response = _anthropic.messages.create(
        model=ROUTER_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": build_router_prompt(oracle_export)}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if model wraps in ```json
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    return json.loads(raw)

# ─── Sorter Agent (one per domain) ─────────────────────────────────────────────

async def run_sorter(
    domain: dict,
    content: str,
    session_date: str,
    client: httpx.AsyncClient,
) -> DomainResult:
    """Write routed content for one domain to its GitHub repo."""
    repo = domain["repo"]
    domain_id = domain["id"]

    if not content or not content.strip():
        return DomainResult(domain=domain_id, repo=repo, status="skipped", reason="no content")

    try:
        # File path: YYYY/MM/YYYY-MM-DD-{domain}.md
        year, month = session_date[:4], session_date[5:7]
        filename = f"{session_date}-{domain_id}.md"
        path = f"{year}/{month}/{filename}"

        existing, sha = await gh_get_file(client, repo, path)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        if existing:
            separator = f"\n\n---\n\n## Appended — {now_utc}\n\n"
            new_content = existing + separator + content
            action = "append"
            commit_msg = f"MISTR KB: append {domain_id} ({session_date})"
        else:
            header = (
                f"# {domain_id.replace('-', ' ').title()} — {session_date}\n\n"
                f"*Auto-generated by MISTR KB Sorter Pipeline · {now_utc}*\n\n---\n\n"
            )
            new_content = header + content
            action = "create"
            sha = None
            commit_msg = f"MISTR KB: {domain_id} session notes {session_date}"

        await gh_write_file(client, repo, path, new_content, commit_msg, sha)
        return DomainResult(domain=domain_id, repo=repo, status="written", path=path, action=action)

    except httpx.HTTPStatusError as e:
        return DomainResult(domain=domain_id, repo=repo, status="error", error=f"HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        return DomainResult(domain=domain_id, repo=repo, status="error", error=str(e)[:300])

# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/sort", response_model=SortResponse)
async def sort_oracle_export(request: SortRequest):
    """
    Receive Oracle Export markdown → route with claude-haiku-4-5 → write to 12 KB repos in parallel.
    """
    if not request.content or len(request.content.strip()) < 100:
        raise HTTPException(status_code=400, detail="content must be at least 100 characters")

    session_date = request.session_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Route with haiku
    try:
        routed = route_content(request.content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Router returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {e}")

    # 2. Run 12 sorter agents in parallel
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
        tasks = [
            run_sorter(domain, routed.get(domain["id"], ""), session_date, http)
            for domain in KB_DOMAINS
        ]
        results: list[DomainResult] = await asyncio.gather(*tasks)  # type: ignore[assignment]

    written  = sum(1 for r in results if r.status == "written")
    skipped  = sum(1 for r in results if r.status == "skipped")
    errored  = sum(1 for r in results if r.status == "error")

    return SortResponse(
        status="completed" if errored == 0 else "partial",
        session_date=session_date,
        files_written=written,
        domains_skipped=skipped,
        domains_errored=errored,
        results=results,
    )


@app.get("/kb-status")
async def kb_status():
    """Return .md file counts per KB repo."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as http:
        tasks = [gh_count_all_md(http, d["repo"]) for d in KB_DOMAINS]
        counts = await asyncio.gather(*tasks, return_exceptions=True)

    domains_status = {}
    for i, domain in enumerate(KB_DOMAINS):
        result = counts[i]
        if isinstance(result, Exception):
            domains_status[domain["id"]] = {"repo": domain["repo"], "md_files": -1, "error": str(result)}
        else:
            domains_status[domain["id"]] = {"repo": domain["repo"], "md_files": result}

    total = sum(v.get("md_files", 0) for v in domains_status.values() if isinstance(v.get("md_files"), int) and v["md_files"] >= 0)
    return {
        "org": GITHUB_ORG,
        "total_md_files": total,
        "domains": domains_status,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kb-sorter-pipeline", "model": ROUTER_MODEL, "domains": len(KB_DOMAINS)}
