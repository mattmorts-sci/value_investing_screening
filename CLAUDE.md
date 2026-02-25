# Algorithmic Investing Pipeline

## Environment

- Python 3.12.8 (pyenv)
- Working directory: ~/projects/Pers/value_investing_screening
<!-- - Secrets in `.env` (see `.env.example`) -->

## Project Documents

<!-- - **Architecture:** [path, e.g. docs/ARCHITECTURE.md] -->
- **Contracts:** `Pers/algorithmic-investing/contracts.py`
<!-- - **Design docs:** `Pers/algorithmic-investing/TECHNICAL_REFERENCE.md`, `Pers/algorithmic-investing/USER_GUIDE.md`  -->
<!-- - **Task tracking:** [path, e.g. REBUILD_PLAN.md] -->

<!-- All design documents are in path Pers/algorithmic-investing/docs  -->

## Code documentation
The following must be kept up to date:
- TECHNICAL_REFERENCE.md - this document is a technical document for agent reference
- USER_GUIDE.md - this is a detailed, plan English user reference

## Agent Team Conventions

- **Parallel implementation:** teammates own files, not layers. One
  teammate per module/block. File ownership must be explicit and
  non-overlapping before work begins.
- **Communication:** teammates communicate via SendMessage (inbox-based
  messaging). They do NOT share context or memory — only messages and
  the shared task list.
- **Sequential review:** reviewer teammates are instructed to use only
  Read, Grep, and Glob (soft enforcement). A clean context window with
  no implementation history is the primary isolation mechanism.
- Decisions that need human input are surfaced immediately, not batched.
- The lead must not use Edit, Write, or Bash directly in team mode.
  (Delegate mode is currently broken — do not use it. This is
  instruction-enforced only.)
- All teammates must be shut down before the lead completes.

## Compact Instructions

When compacting, preserve:
- Current task and its mode (solo/build/team)
- Active teammate names and their assignments (if team mode)
- Unresolved issues and pending decisions
- Which files have been modified in this session
- Any accumulated rules from this conversation

Discard:
- Verbose tool outputs (lint results, test output) — keep only failures
- File contents that can be re-read from disk
- Completed subtask details (keep only the outcome)

## Coding Workflow

Skills are in `.claude/skills/`. 

Human checkpoints:
1. Design decisions (before implementation begins)
2. Issues requiring judgement (NEEDS_DECISION items, surfaced live)
3. Pre-commit review (staged diff summary via `/commit`)

Inline linting (ruff) runs automatically via PostToolUse hook during
all modes. Full pre-commit checks (ruff, mypy, pytest) are handled
by `/commit` at the end of the workflow.

## Commands

- Run tests: `pytest -v`
- Lint: `ruff check .`
- Fix lint: `ruff check --fix .`
- Type check: `mypy .`
- Pre-commit sequence: `ruff check . && mypy . && pytest -v`

## Data Sources

- **SQLite:** Local database for cached/downloaded data. Never modify raw data files.
No sqlite3 CLI. Use Python instead.1

## Code Style

See `STYLE_GUIDE.md` for full conventions. Critical rules repeated here:

- All configurable values imported from central config — never hardcoded in source files
- `logging` module only — no `print()` statements in source files
- Type hints on all function signatures (enforced by mypy)
- Docstrings on all public functions and classes
- No YAML — config is Python dataclasses only

## Error Handling

- **Data integrity issues:** Hard failure. Raise exceptions on ambiguous, missing, or contradictory data. No silent defaults.
- **Expected operational conditions:** Handle gracefully where ARCHITECTURE.md explicitly defines the behaviour. Examples: skip windows with insufficient samples (log warning), fill NaN with 0 (log explicitly), skip windows where all models fail (log error).
- **No data leakage.** Training data must never contain future information.
- **Ask, don't assume** on financial logic.

## Current Status

live data testing of the full pipeline.

## Accumulated Rules

- **No file changes without explicit user instruction.** No creating, deleting, or modifying files unless told to.
- Wait for user decisions. Do not interpret analysis as authorisation to act.
- Do only what is asked. Answer questions directly; do not infer next steps.
- No motivational language. Describe, don't coach.
- Answer direct questions directly before doing other work.
- Prompt the user before fixing critical/high/medium issues. Return to planning where required.
- Australian English.
- One block at a time. Only read documents needed for the current block.
- One question or decision at a time.
- Update CONTRACTS.md, pyproject.toml etc. at end of each build phase.
- Never create a config parameter derivable from an existing one.
- No sycophancy. Assess independently; disagree with reasoning when warranted.
- Direct, factual language. No hedging, no filler.
- After user has given direction backbrief BEFORE starting a task