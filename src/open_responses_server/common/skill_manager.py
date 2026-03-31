import os
import asyncio
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, ClassVar, Optional

import yaml

from .config import SKILLS_ENABLED, SKILLS_DIR, SKILLS_EXEC_TIMEOUT, logger

# Maximum characters of SKILL.md instructions to include in tool description
_MAX_INSTRUCTIONS_LENGTH = 2000


def _parse_skill_md(content: str) -> Optional[dict]:
    """Parse SKILL.md content into frontmatter dict + instructions body.

    Returns a dict with 'name', 'description', and 'instructions' keys,
    or None if parsing fails.
    """
    content = content.strip()
    if not content.startswith("---"):
        return None

    # Find closing --- delimiter
    end = content.find("---", 3)
    if end == -1:
        return None

    frontmatter_text = content[3:end].strip()
    instructions = content[end + 3:].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return None

    if not isinstance(frontmatter, dict):
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description", "")
    if not name:
        return None

    return {
        "name": str(name),
        "description": str(description),
        "instructions": instructions,
    }


@dataclass
class Skill:
    """Represents a single discovered agent skill."""
    name: str
    description: str
    instructions: str
    base_dir: Path
    scripts_dir: Path
    references_dir: Path
    available_scripts: List[str] = field(default_factory=list)


class SkillManager:
    """Manages agent skill discovery, caching, and script execution.

    Security: This feature is OFF by default (SKILLS_ENABLED=false).
    When enabled, only scripts inside a skill's scripts/ directory can be
    executed. Path traversal is prevented via resolve() + prefix checks.
    No shell=True is ever used.
    """
    _instance: ClassVar[Optional["SkillManager"]] = None

    def __init__(self):
        self.skills: List[Skill] = []
        self.skill_functions_cache: List[Dict[str, Any]] = []
        self._skill_name_mapping: Dict[str, Skill] = {}

    @classmethod
    def get_instance(cls) -> "SkillManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def startup_skills(self) -> None:
        """Scan SKILLS_DIR, parse each skill, and populate caches.

        No-op when SKILLS_ENABLED is False or SKILLS_DIR is empty.
        """
        if not SKILLS_ENABLED:
            logger.info("[SKILLS-STARTUP] Agent skills are disabled (SKILLS_ENABLED=false)")
            return

        if not SKILLS_DIR:
            logger.warning("[SKILLS-STARTUP] SKILLS_ENABLED is true but SKILLS_DIR is empty")
            return

        skills_path = Path(SKILLS_DIR)
        if not skills_path.is_dir():
            logger.warning(f"[SKILLS-STARTUP] SKILLS_DIR does not exist or is not a directory: {skills_path}")
            return

        logger.info(f"[SKILLS-STARTUP] Scanning skills directory: {skills_path}")

        for entry in sorted(skills_path.iterdir()):
            if not entry.is_dir():
                continue

            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                logger.debug(f"[SKILLS-STARTUP] Skipping '{entry.name}': no SKILL.md found")
                continue

            try:
                content = await asyncio.to_thread(skill_md.read_text, encoding="utf-8")
                parsed = _parse_skill_md(content)
                if parsed is None:
                    logger.warning(f"[SKILLS-STARTUP] Skipping '{entry.name}': invalid SKILL.md frontmatter")
                    continue

                scripts_dir = entry / "scripts"
                references_dir = entry / "references"

                available_scripts: List[str] = []
                if scripts_dir.is_dir():
                    for script_file in sorted(scripts_dir.iterdir()):
                        if script_file.is_file() and os.access(script_file, os.X_OK):
                            available_scripts.append(script_file.name)

                if not available_scripts:
                    logger.warning(f"[SKILLS-STARTUP] Skill '{parsed['name']}' has no executable scripts in scripts/")
                    continue

                skill = Skill(
                    name=parsed["name"],
                    description=parsed["description"],
                    instructions=parsed["instructions"],
                    base_dir=entry.resolve(),
                    scripts_dir=scripts_dir.resolve() if scripts_dir.is_dir() else scripts_dir,
                    references_dir=references_dir.resolve() if references_dir.is_dir() else references_dir,
                    available_scripts=available_scripts,
                )
                self.skills.append(skill)
                logger.info(
                    f"[SKILLS-STARTUP] Registered skill '{skill.name}' "
                    f"with {len(available_scripts)} scripts: {available_scripts}"
                )

            except Exception as e:
                logger.error(f"[SKILLS-STARTUP] Error loading skill from '{entry.name}': {e}")

        # Build caches
        self._build_caches()
        logger.info(
            f"[SKILLS-STARTUP] Loaded {len(self.skills)} skills, "
            f"{len(self.skill_functions_cache)} tool definitions"
        )

    async def shutdown_skills(self) -> None:
        """Clean up skill manager state."""
        logger.info(f"[SKILLS-SHUTDOWN] Clearing {len(self.skills)} skills")
        self.skills.clear()
        self.skill_functions_cache.clear()
        self._skill_name_mapping.clear()

    def _build_caches(self) -> None:
        """Build tool definitions and name mapping from discovered skills."""
        self.skill_functions_cache.clear()
        self._skill_name_mapping.clear()

        for skill in self.skills:
            tool_name = f"skill__{skill.name}"

            # Build description with truncated instructions
            instructions_preview = skill.instructions[:_MAX_INSTRUCTIONS_LENGTH]
            if len(skill.instructions) > _MAX_INSTRUCTIONS_LENGTH:
                instructions_preview += "\n... (truncated)"

            description_parts = [skill.description]
            if skill.available_scripts:
                description_parts.append(f"Available scripts: {', '.join(skill.available_scripts)}")
            if instructions_preview:
                description_parts.append(instructions_preview)

            tool_def = {
                "name": tool_name,
                "description": "\n\n".join(description_parts),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": f"Script to run. Must be one of: {', '.join(skill.available_scripts)}",
                            "enum": skill.available_scripts,
                        },
                        "args": {
                            "type": "string",
                            "description": "Command-line arguments to pass to the script",
                            "default": "",
                        },
                    },
                    "required": ["script"],
                },
            }

            self.skill_functions_cache.append(tool_def)
            self._skill_name_mapping[tool_name] = skill

    def get_skill_tools(self) -> List[Dict[str, Any]]:
        """Returns the cached list of skill tool definitions."""
        return self.skill_functions_cache

    def is_skill_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to a registered skill."""
        return tool_name in self._skill_name_mapping

    async def execute_skill_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a skill script and return its output.

        Raises RuntimeError if the tool is not found, the script is invalid,
        or execution fails.
        """
        skill = self._skill_name_mapping.get(tool_name)
        if skill is None:
            raise RuntimeError(f"Skill tool '{tool_name}' not found")

        script_name = arguments.get("script", "")
        args_str = arguments.get("args", "")

        if not script_name:
            raise RuntimeError(f"No 'script' argument provided for skill '{skill.name}'")

        # Path validation: prevent directory traversal
        script_path = (skill.scripts_dir / script_name).resolve()
        scripts_dir_resolved = skill.scripts_dir.resolve()

        if not str(script_path).startswith(str(scripts_dir_resolved) + os.sep) and script_path != scripts_dir_resolved:
            logger.error(
                f"[SKILLS-EXEC] Path traversal attempt blocked: "
                f"'{script_name}' resolves outside scripts dir"
            )
            raise RuntimeError(f"Invalid script path: '{script_name}'")

        if not script_path.is_file():
            raise RuntimeError(
                f"Script '{script_name}' not found in skill '{skill.name}'. "
                f"Available: {skill.available_scripts}"
            )

        if not os.access(script_path, os.X_OK):
            raise RuntimeError(f"Script '{script_name}' is not executable")

        # Build command
        cmd = [str(script_path)]
        if args_str:
            cmd.extend(shlex.split(args_str))

        logger.info(
            f"[SKILLS-EXEC] Executing skill '{skill.name}' "
            f"script '{script_name}' with args: {args_str}"
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(skill.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=SKILLS_EXEC_TIMEOUT,
            )

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if process.returncode != 0:
                error_msg = (
                    f"Script '{script_name}' exited with code {process.returncode}"
                )
                if stderr_text:
                    error_msg += f"\nstderr: {stderr_text}"
                if stdout_text:
                    error_msg += f"\nstdout: {stdout_text}"
                logger.warning(f"[SKILLS-EXEC] {error_msg}")
                raise RuntimeError(error_msg)

            logger.info(
                f"[SKILLS-EXEC] Skill '{skill.name}' script '{script_name}' "
                f"completed successfully ({len(stdout_text)} chars output)"
            )
            return stdout_text if stdout_text else "(no output)"

        except asyncio.TimeoutError:
            logger.error(
                f"[SKILLS-EXEC] Script '{script_name}' in skill '{skill.name}' "
                f"timed out after {SKILLS_EXEC_TIMEOUT}s"
            )
            # Kill the process on timeout
            try:
                process.kill()  # type: ignore[possibly-undefined]
                await process.wait()  # type: ignore[possibly-undefined]
            except Exception:  # nosec B110 - best-effort cleanup
                logger.debug("Failed to kill timed-out process")
            raise RuntimeError(
                f"Script '{script_name}' timed out after {SKILLS_EXEC_TIMEOUT}s"
            )
        except Exception as e:
            logger.error(f"[SKILLS-EXEC] Error executing script '{script_name}': {e}")
            raise RuntimeError(f"Error executing script '{script_name}': {e}")


# Singleton instance
skill_manager = SkillManager.get_instance()
