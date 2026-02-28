# Value Investing Screening Pipeline

## Environment

- Python 3.12.8 (pyenv)
- Working directory: ~/projects/Pers/value_investing_screening
<!-- - Secrets in `.env` (see `.env.example`) -->

## Project Documents

- **Technical reference:** `TECHNICAL_REFERENCE.md`
- **User guide:** `USER_GUIDE.md`

## Code Documentation
The following must be kept up to date:
- `TECHNICAL_REFERENCE.md` — technical reference for agent use (architecture, interfaces, data flow)
- `USER_GUIDE.md` — detailed, plain English user reference

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
- Current task and its mode 
- Active teammate names and their assignments (if team mode)
- Unresolved issues and pending decisions
- Which files have been modified in this session
- Any accumulated rules from this conversation

Discard:
- Verbose tool outputs (lint results, test output) — keep only failures
- File contents that can be re-read from disk
- Completed subtask details (keep only the outcome)

## Coding Workflow

Skills are in `.claude/skills/`. Available skills:

- `/analyse` — Legacy code analysis. Produces structured inventory of
  modules, dependencies, data flow, and technical debt. Read-only.
  Feeds into `/design`.
- `/design` — Refactoring design brief. Covers legacy summary, target
  design, keep/rewrite/delete classification, migration strategy.
  Solo mode, iterative human input.
- `/coord` — Task coordination. Selects mode (solo/build/team) and
  manages the refactoring workflow through four gates:
  analyse → design → implement → verify.
- `/verify` — Code and documentation review. Build, Team, and Accuracy
  modes with adversarial personas and consensus rule.
- `/commit` — Pre-commit checks (ruff → mypy → pytest) and git commit.
  Use `--docs` for documentation-only commits.

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

Note: dev tools (ruff, mypy, pytest) not yet installed. Install before
first `/commit`.

## Data Sources

- **yfinance:** Real-time market data (prices, financials) via Yahoo Finance API
- **CSV:** Static reference data in `input/` directory. Never modify raw data files.

## Code Style

- All configurable values imported from central config — never hardcoded in source files
- `logging` module only — no `print()` statements in source files
- Type hints on all function signatures (enforced by mypy)
- Docstrings on all public functions and classes
- No YAML — config is Python dataclasses only

## Error Handling

- **Data integrity issues:** Hard failure. Raise exceptions on ambiguous, missing, or contradictory data. No silent defaults.
- **Expected operational conditions:** Handle gracefully where documented. Examples: skip tickers with insufficient data (log warning), handle API rate limits with retry/backoff.
- **Ask, don't assume** on financial logic.

## Current Status

Preparing to refactor legacy code. Legacy code is in `LEGACY/`.

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
- Update TECHNICAL_REFERENCE.md, USER_GUIDE.md etc. at end of each build phase.
- Never create a config parameter derivable from an existing one.
- No sycophancy. Assess independently; disagree with reasoning when warranted.
- Direct, factual language. No hedging, no filler.
- "Backbrief" means: restate your understanding of the task and stop. After user has given direction backbrief, wait for user approaval BEFORE starting a task
- Documentation records what IS, not what ISN'T. Do not document excluded options, or how its changed from a previous version.
- Always recommend fixing known issues immediately. Present findings and wait for user decision before acting.