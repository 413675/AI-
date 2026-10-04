"""Skill 服务：扫描项目 skills/ 目录下的技能包，供智能体按需加载执行。

技能包格式：skills/<目录名>/SKILL.md，文件头部用 YAML frontmatter 声明
name / description，正文为 LLM 执行该技能时的完整指令。
"""
from pathlib import Path


class SkillService:
    def __init__(self, skills_dir: Path | None = None):
        # 默认：backend/AI-tablepet/skills/
        self.skills_dir = skills_dir or Path(__file__).resolve().parent.parent.parent / "skills"

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """解析 --- 包裹的 frontmatter（简单键值，避免引入 yaml 依赖）"""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                meta = {}
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
                return meta, parts[2].strip()
        return {}, text

    def list_skills(self) -> list[dict]:
        """列出全部可用技能：[{name, description, dir}]"""
        skills = []
        if not self.skills_dir.exists():
            return skills
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = md.read_text(encoding="utf-8-sig")
            except OSError:
                continue
            meta, _ = self._parse_frontmatter(text)
            skills.append(
                {
                    "name": meta.get("name") or md.parent.name,
                    "description": meta.get("description", ""),
                    "dir": md.parent.name,
                }
            )
        return skills

    def load_skill(self, name: str) -> str:
        """按技能名（frontmatter name 或目录名）加载完整 SKILL.md 指令"""
        for s in self.list_skills():
            if name in (s["name"], s["dir"]):
                return (self.skills_dir / s["dir"] / "SKILL.md").read_text(encoding="utf-8-sig")
        available = "、".join(s["name"] for s in self.list_skills()) or "无"
        return f"未找到技能「{name}」。可用技能: {available}"


skill_service = SkillService()