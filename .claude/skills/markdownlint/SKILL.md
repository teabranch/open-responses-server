---
name: markdownlint
description: >
  Lint and auto-fix markdown files in the open-responses-server repo using
  markdownlint-cli2. Use when: editing markdown docs, before committing doc
  changes, or when the user says "lint docs", "check markdown", or "fix
  markdown".
---

# Markdownlint Skill

Lint all documentation markdown files and optionally auto-fix issues.

## Quick Usage

### Check all docs

```bash
bash .claude/skills/markdownlint/scripts/lint-docs.sh
```

### Auto-fix fixable issues

```bash
bash .claude/skills/markdownlint/scripts/lint-docs.sh --fix
```

### Check specific files

```bash
bash .claude/skills/markdownlint/scripts/lint-docs.sh docs/events-and-tool-handling.md
```

## What Gets Linted

By default the script lints:

- `docs/*.md` — top-level documentation
- `index.md` — Jekyll home page
- `CLAUDE.md` — repo guidance
- `.claude/skills/**/*.md` — skill definitions

Excluded by default:

- `docs/plan/` — historical planning archives
- `docs/prompts/` — prompt templates
- `docs/pip-publish-instructions.md` — legacy doc
- `docs/using-uv.md` — legacy doc

## Configuration

The script uses whichever config is found first:

1. Project-level `.markdownlint-cli2.yaml` (in repo root)
2. Global `~/.markdownlint-cli2.yaml`
3. markdownlint built-in defaults

The global config disables:

- **MD013** (line length) — too noisy for content-heavy docs and tables
- **MD060** (table pipe spacing) — stylistic preference

## Workflow

After editing any `.md` file:

1. Run `bash .claude/skills/markdownlint/scripts/lint-docs.sh`
2. If issues found, run with `--fix` to auto-fix what can be fixed
3. Manually fix remaining issues
4. Re-run to verify clean

## Common Rules

| Rule | What It Checks |
| --- | --- |
| MD001 | Heading levels increment by one |
| MD004 | Consistent unordered list style (dashes) |
| MD009 | No trailing spaces |
| MD022 | Blank lines around headings |
| MD025 | Single H1 per document (use front matter title) |
| MD031 | Blank lines around fenced code blocks |
| MD032 | Blank lines around lists |
| MD040 | Fenced code blocks should have a language |
| MD047 | Files end with a single newline |
