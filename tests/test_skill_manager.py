"""
Tests for SkillManager: skill discovery, caching, path validation, and execution.
"""
import os
import stat
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from open_responses_server.common.skill_manager import (
    SkillManager,
    Skill,
    _parse_skill_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit tests
---

## Instructions

Run the echo script to test output.
"""

MINIMAL_SKILL_MD = """\
---
name: minimal
description: Minimal skill
---
"""

NO_NAME_SKILL_MD = """\
---
description: Missing name field
---

Some instructions.
"""

INVALID_YAML_SKILL_MD = """\
---
name: [invalid yaml
  bad: {{
---
"""


def _create_skill_dir(tmp_path: Path, name: str, skill_md: str,
                      scripts: dict | None = None) -> Path:
    """Create a skill directory with SKILL.md and optional scripts."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(skill_md)

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    references_dir = skill_dir / "references"
    references_dir.mkdir()

    if scripts:
        for script_name, content in scripts.items():
            script_path = scripts_dir / script_name
            script_path.write_text(content)
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    return skill_dir


# ---------------------------------------------------------------------------
# _parse_skill_md
# ---------------------------------------------------------------------------

class TestParseSkillMd:
    """Tests for the SKILL.md frontmatter parser."""

    def test_valid_frontmatter(self):
        result = _parse_skill_md(VALID_SKILL_MD)
        assert result is not None
        assert result["name"] == "test-skill"
        assert result["description"] == "A test skill for unit tests"
        assert "Run the echo script" in result["instructions"]

    def test_minimal_frontmatter(self):
        result = _parse_skill_md(MINIMAL_SKILL_MD)
        assert result is not None
        assert result["name"] == "minimal"
        assert result["instructions"] == ""

    def test_missing_name_returns_none(self):
        result = _parse_skill_md(NO_NAME_SKILL_MD)
        assert result is None

    def test_invalid_yaml_returns_none(self):
        result = _parse_skill_md(INVALID_YAML_SKILL_MD)
        assert result is None

    def test_no_frontmatter_returns_none(self):
        result = _parse_skill_md("Just plain text, no frontmatter")
        assert result is None

    def test_missing_closing_delimiter_returns_none(self):
        result = _parse_skill_md("---\nname: test\n")
        assert result is None


# ---------------------------------------------------------------------------
# SkillManager singleton
# ---------------------------------------------------------------------------

class TestSkillManagerSingleton:

    def test_get_instance_returns_same_object(self):
        # Reset singleton for test isolation
        SkillManager._instance = None
        a = SkillManager.get_instance()
        b = SkillManager.get_instance()
        assert a is b
        SkillManager._instance = None  # cleanup


# ---------------------------------------------------------------------------
# SkillManager startup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSkillManagerStartup:

    async def test_startup_disabled_is_noop(self):
        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", False):
            await mgr.startup_skills()
        assert mgr.skills == []
        assert mgr.skill_functions_cache == []

    async def test_startup_empty_dir_is_noop(self):
        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", ""):
            await mgr.startup_skills()
        assert mgr.skills == []

    async def test_startup_missing_dir_logs_warning(self, tmp_path):
        mgr = SkillManager()
        missing = str(tmp_path / "nonexistent")
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", missing):
            await mgr.startup_skills()
        assert mgr.skills == []

    async def test_startup_scans_directory(self, tmp_path):
        _create_skill_dir(tmp_path, "my-skill", VALID_SKILL_MD,
                          scripts={"echo.sh": "#!/bin/bash\necho hello"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        assert len(mgr.skills) == 1
        assert mgr.skills[0].name == "test-skill"
        assert mgr.skills[0].available_scripts == ["echo.sh"]

    async def test_startup_skips_malformed_skill_md(self, tmp_path):
        _create_skill_dir(tmp_path, "good-skill", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho ok"})
        _create_skill_dir(tmp_path, "bad-skill", NO_NAME_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho bad"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        assert len(mgr.skills) == 1
        assert mgr.skills[0].name == "test-skill"

    async def test_startup_skips_no_scripts(self, tmp_path):
        # Create skill dir with SKILL.md but no executable scripts
        skill_dir = tmp_path / "empty-scripts"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "references").mkdir()

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        assert len(mgr.skills) == 0


# ---------------------------------------------------------------------------
# Tool cache and lookup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSkillToolCache:

    async def test_get_skill_tools_returns_correct_shape(self, tmp_path):
        _create_skill_dir(tmp_path, "demo", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho demo"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        tools = mgr.get_skill_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert tool["name"] == "skill__test-skill"
        assert "A test skill" in tool["description"]
        assert tool["parameters"]["type"] == "object"
        assert "script" in tool["parameters"]["properties"]
        assert tool["parameters"]["properties"]["script"]["enum"] == ["run.sh"]
        assert "args" in tool["parameters"]["properties"]
        assert tool["parameters"]["required"] == ["script"]

    async def test_is_skill_tool_true(self, tmp_path):
        _create_skill_dir(tmp_path, "demo", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho x"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        assert mgr.is_skill_tool("skill__test-skill") is True

    async def test_is_skill_tool_false(self):
        mgr = SkillManager()
        assert mgr.is_skill_tool("skill__nonexistent") is False
        assert mgr.is_skill_tool("some_mcp_tool") is False


# ---------------------------------------------------------------------------
# Skill execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSkillExecution:

    async def test_execute_skill_tool_success(self, tmp_path):
        _create_skill_dir(tmp_path, "echo-skill", VALID_SKILL_MD,
                          scripts={"echo.sh": "#!/bin/bash\necho 'hello world'"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        result = await mgr.execute_skill_tool(
            "skill__test-skill",
            {"script": "echo.sh", "args": ""}
        )
        assert result == "hello world"

    async def test_execute_skill_tool_with_args(self, tmp_path):
        _create_skill_dir(tmp_path, "args-skill", VALID_SKILL_MD,
                          scripts={"greet.sh": "#!/bin/bash\necho \"Hello $1\""})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        result = await mgr.execute_skill_tool(
            "skill__test-skill",
            {"script": "greet.sh", "args": "World"}
        )
        assert result == "Hello World"

    async def test_execute_skill_tool_path_traversal_rejected(self, tmp_path):
        _create_skill_dir(tmp_path, "safe-skill", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho safe"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        with pytest.raises(RuntimeError, match="Invalid script path"):
            await mgr.execute_skill_tool(
                "skill__test-skill",
                {"script": "../../etc/passwd", "args": ""}
            )

    async def test_execute_skill_tool_nonexistent_script(self, tmp_path):
        _create_skill_dir(tmp_path, "missing-script", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho x"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        with pytest.raises(RuntimeError, match="not found"):
            await mgr.execute_skill_tool(
                "skill__test-skill",
                {"script": "nonexistent.sh", "args": ""}
            )

    async def test_execute_skill_tool_nonzero_exit(self, tmp_path):
        _create_skill_dir(tmp_path, "fail-skill", VALID_SKILL_MD,
                          scripts={"fail.sh": "#!/bin/bash\necho 'error output' >&2\nexit 1"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        with pytest.raises(RuntimeError, match="exited with code 1") as exc_info:
            await mgr.execute_skill_tool(
                "skill__test-skill",
                {"script": "fail.sh", "args": ""}
            )
        assert "error output" in str(exc_info.value)

    async def test_execute_skill_tool_timeout(self, tmp_path):
        _create_skill_dir(tmp_path, "slow-skill", VALID_SKILL_MD,
                          scripts={"slow.sh": "#!/bin/bash\nsleep 60"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)), \
             patch("open_responses_server.common.skill_manager.SKILLS_EXEC_TIMEOUT", 1):
            await mgr.startup_skills()

        with pytest.raises(RuntimeError, match="timed out"):
            await mgr.execute_skill_tool(
                "skill__test-skill",
                {"script": "slow.sh", "args": ""}
            )

    async def test_execute_skill_tool_not_found(self):
        mgr = SkillManager()
        with pytest.raises(RuntimeError, match="not found"):
            await mgr.execute_skill_tool(
                "skill__nonexistent",
                {"script": "run.sh", "args": ""}
            )

    async def test_execute_skill_tool_no_script_arg(self, tmp_path):
        _create_skill_dir(tmp_path, "no-arg-skill", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho x"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        with pytest.raises(RuntimeError, match="No 'script' argument"):
            await mgr.execute_skill_tool(
                "skill__test-skill",
                {"args": "some args"}
            )


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSkillManagerShutdown:

    async def test_shutdown_clears_state(self, tmp_path):
        _create_skill_dir(tmp_path, "cleanup-skill", VALID_SKILL_MD,
                          scripts={"run.sh": "#!/bin/bash\necho ok"})

        mgr = SkillManager()
        with patch("open_responses_server.common.skill_manager.SKILLS_ENABLED", True), \
             patch("open_responses_server.common.skill_manager.SKILLS_DIR", str(tmp_path)):
            await mgr.startup_skills()

        assert len(mgr.skills) == 1
        await mgr.shutdown_skills()
        assert mgr.skills == []
        assert mgr.skill_functions_cache == []
        assert mgr._skill_name_mapping == {}
