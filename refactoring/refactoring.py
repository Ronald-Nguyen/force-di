import os
from pyexpat import model
import re
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from unittest import result
import difflib
import time
import json
import xml.etree.ElementTree as ET 
'''
python refactoring/refactoring.py
python refactoring/refactoring.py --all-refactorings
python refactoring/refactoring.py --refactoring rename
'''


REFACTORINGS = [
    #"coc_reduktion",
    #"getter_setter",
    "guard_clauses",
    "inline_variable",
    #"rename",
    "strategy_pattern",
]
REFACTORING_BASE_DIR = "refactoring"
DEFAULT_REFACTORING = "coc_reduktion" \
""
RESULT_PATH = "_result_"
PATH = 'force-di/main'
ITERATIONS = 10
GEMMA = 'gemma-3-27b-it'
GEMINI3 = 'gemini-3-pro-preview'
GEMINI2 = 'gemini-2.5-flash'
LLAMA = 'llama-3.3-70b-versatile'
MISTRAL = 'mistral-large-2512'
CODESTRAL = 'codestral-2501'
NVIDIA = 'nvidia/llama-3.1-nemotron-ultra-253b-v1'
MODEL_OLLAMA = 'devstral-2_123b-cloud'
MODEL_GROQ = LLAMA
MODEL_GEMINI = GEMINI3
MODEL_MISTRAL = CODESTRAL
MODEL_NVIDIA = NVIDIA
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY2')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
LLM_API_KEY = MISTRAL_API_KEY    
client = None
MODEL = None

if LLM_API_KEY == MISTRAL_API_KEY:
    from mistralai import Mistral
    MODEL = MODEL_MISTRAL
    try:
        client = Mistral(api_key=LLM_API_KEY)
        print("Mistral API Key aus Umgebungsvariable geladen")
    except Exception as e:
        print(f"Fehler beim Laden des API-Keys: {e}")
        exit(1)
elif LLM_API_KEY == GEMINI_API_KEY:
    from google import genai
    MODEL = MODEL_GEMINI
    try:
        client = genai.Client(api_key=LLM_API_KEY)
        print("Gemini API Key aus Umgebungsvariable geladen")
    except Exception as e:
        print(f"Fehler beim Laden des API-Keys: {e}")
        exit(1)
elif LLM_API_KEY == GROQ_API_KEY:
    from groq import Groq
    MODEL = MODEL_GROQ
    try:
        client = Groq(api_key=LLM_API_KEY)
        print("Groq API Key aus Umgebungsvariable geladen")
    except Exception as e:
        print(f"Fehler beim Laden des API-Keys: {e}")
        exit(1)
elif LLM_API_KEY == NVIDIA_API_KEY:
    from openai import OpenAI
    MODEL = MODEL_NVIDIA
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=LLM_API_KEY
        )
        print("NVIDIA API Key aus Umgebungsvariable geladen")
    except Exception as e:
        print(f"Fehler beim Laden des API-Keys: {e}")
        exit(1)

# Sanitize MODEL for use in file paths (replace / with -)
MODEL_SAFE = MODEL.replace('/', '-') if MODEL else None

parser = argparse.ArgumentParser(description="Projektpfad angeben")
parser.add_argument("--project-path", type=str, default=PATH, help="Pfad des Projekts")
parser.add_argument("--all-refactorings", action="store_true",
                    help="Wenn gesetzt: führt alle Refactorings nacheinander aus.")
parser.add_argument("--refactoring", type=str, default=DEFAULT_REFACTORING,
                    choices=REFACTORINGS,
                    help="Welches Refactoring ausgeführt werden soll (wenn --all-refactorings nicht gesetzt ist).")
args = parser.parse_args()


def _resolve_file_hint(project_root: Path, file_hint: str, changed_rel_paths: list[str]) -> str | None:
    """
    Resolves file_hint from prompt to a project-relative path.
    Priority:
    1) exact match in changed_rel_paths
    2) basename match in changed_rel_paths
    3) global scan for basename in project (first hit)
    Returns normalized rel path with forward slashes, or None.
    """
    if not file_hint:
        return None

    hint = str(Path(file_hint)).replace("\\", "/")

    # 1) exact match
    for rel in changed_rel_paths:
        if rel.replace("\\", "/") == hint:
            return rel.replace("\\", "/")

    # 2) basename match among changed files
    hint_base = Path(hint).name.lower()
    for rel in changed_rel_paths:
        if Path(rel).name.lower() == hint_base:
            return rel.replace("\\", "/")

    # 3) global scan
    for fp in _iter_apex_files(project_root):
        if fp.name.lower() == hint_base:
            return str(fp.relative_to(project_root)).replace("\\", "/")

    return None

def _scan_files_for_regex(project_root: Path, rel_paths: list[str], pattern: str) -> list[str]:
    """
    Scans ONLY the given rel_paths for pattern (comments stripped).
    Returns list of rel paths where pattern matches.
    """
    rx = re.compile(pattern)
    hits: list[str] = []
    pr = project_root.resolve()

    for rel in rel_paths:
        p = (pr / Path(rel)).resolve()
        try:
            p.relative_to(pr)
        except Exception:
            continue
        if not p.exists() or p.is_dir():
            continue

        content = _strip_apex_comments(_read_text_best_effort(p))
        if rx.search(content):
            hits.append(str(Path(rel)).replace("\\", "/"))
    return hits

def format_pmd_metrics_summary(pmd_before: dict, pmd_after: dict) -> str:
    """
    Returns a short, single-line summary for the global summary file.
    Example: "PMD:ok CoCΔ(total=-3, prod=-3, test=0)"
    """
    if not pmd_before or not pmd_after:
        return "PMD:n/a"

    if not pmd_before.get("ok") or not pmd_after.get("ok"):
        err_b = pmd_before.get("error")
        err_a = pmd_after.get("error")
        if err_b or err_a:
            return "PMD:fail"
        return "PMD:fail"

    b = pmd_before.get("summary", {})
    a = pmd_after.get("summary", {})

    def _get(cat: str, key: str, src: dict) -> int:
        return int((src.get(cat, {}).get(key, 0) or 0))

    d_total = _get("total", "complexity", a) - _get("total", "complexity", b)
    d_prod = _get("prod", "complexity", a) - _get("prod", "complexity", b)
    d_test = _get("test", "complexity", a) - _get("test", "complexity", b)

    return f"PMD:ok CoCΔ(total={d_total}, prod={d_prod}, test={d_test})"

def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

def _normalize_lines_ignore_whitespace_and_blanklines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for line in text.split("\n"):
        normalized = re.sub(r"\s+", "", line)
        if normalized == "":
            continue
        out.append(normalized)
    return out

def build_diff_between_backup_and_refactored(
    backup_dir: Path,
    project_src: Path,
    snapshot_files: dict[str, str],
) -> tuple[bool, str]:
    diffs: list[str] = []
    has_changes = False

    rel_paths = sorted({str(Path(p)) for p in snapshot_files.keys()})
    for rel in rel_paths:
        rel_path = Path(rel)

        if any(part == "tests" for part in rel_path.parts):
            continue

        orig_path = backup_dir / rel_path
        new_path = project_src / rel_path

        orig_text = _read_text_best_effort(orig_path) if orig_path.exists() else ""
        new_text = _read_text_best_effort(new_path) if new_path.exists() else ""

        orig_norm = _normalize_lines_ignore_whitespace_and_blanklines(orig_text)
        new_norm = _normalize_lines_ignore_whitespace_and_blanklines(new_text)

        if orig_norm == new_norm:
            continue

        has_changes = True
        diff_lines = list(
            difflib.unified_diff(
                orig_norm,
                new_norm,
                fromfile=f"backup/{rel}",
                tofile=f"refactored/{rel}",
                lineterm="",
                n=0,
            )
        )
        if diff_lines:
            diffs.append("\n".join(diff_lines))

    return has_changes, ("\n\n".join(diffs)).strip()

def get_project_structure(project_dir: Path) -> str:
    structure = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'tests', 'pathlib2.egg-info', 'test'}]
        level = root.replace(str(project_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        structure.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.cls') or file.endswith('.trigger'):
                structure.append(f'{subindent}{file}')
    return '\n'.join(structure)

def get_all_apex_files(project_dir: Path) -> str:
    code_block = ""
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'tests', 'pathlib2.egg-info', 'test'}]
        for file in files:
            if "test" in file.lower():
                continue
            if file.endswith('.cls') or file.endswith('.trigger'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    relative_path = file_path.relative_to(project_dir)
                    code_block += f"\n\nFile `{relative_path}`:\n```apex\n"
                    code_block += content + "```\n"
                except Exception as e:
                    print(f"Fehler beim Lesen von {file_path}: {e}")
    return code_block

def parse_ai_response(response_text: str) -> dict:
    files = {}
    if not response_text:
        return files
    pattern = r"File\s+`([^`]+)`:\s*```apex\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    for filename, code in matches:
        files[filename] = code.strip()
    return files

def backup_project(project_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(
        project_dir, backup_dir,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'test', 'tests', 'pathlib2.egg-info')
    )

def restore_project(backup_dir: Path, project_dir: Path) -> None:
    backup_dir = Path(backup_dir).resolve()
    project_dir = Path(project_dir).resolve()

    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup-Verzeichnis nicht gefunden: {backup_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(backup_dir, project_dir, dirs_exist_ok=True)

def apply_changes(project_dir: Path | str, files: dict[str, str]) -> None:
    project_dir = Path(project_dir).resolve()

    for filename, code in files.items():
        file_rel = Path(filename)

        if any(part == 'tests' for part in file_rel.parts):
            continue

        file_path = (project_dir / file_rel).resolve()
        try:
            file_path.relative_to(project_dir)
        except ValueError:
            print(f" {filename} liegt außerhalb von {project_dir}, übersprungen")
            continue

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code, encoding='utf-8')
            print(f" {filename} aktualisiert")
        except Exception as e:
            print(f" Fehler beim Schreiben von {filename}: {e}")

def run_apex_tests(target_org=None):
    env = os.environ.copy()
    env["CI"] = "true"
    env["SF_NO_COLOR"] = "true"
    env["SF_DISABLE_AUTOUPDATE"] = "true"

    deploy_cmd = "sf project deploy start --ignore-conflicts"
    if target_org:
        deploy_cmd += f" --target-org {target_org}"

    print(f" -> Deploying Code...")

    deploy_res = subprocess.run(
        deploy_cmd,
        capture_output=True,
        text=False,
        shell=True,
        env=env
    )

    stdout_str = deploy_res.stdout.decode('utf-8', errors='replace')
    stderr_str = deploy_res.stderr.decode('utf-8', errors='replace')

    if deploy_res.returncode != 0:
        return {
            'level': 0,
            'success': False,
            'stdout': stdout_str,
            'stderr': f"DEPLOY FAILED (Level 0):\n{stderr_str}",
            'returncode': deploy_res.returncode
        }

    test_cmd = "sf apex run test --wait 10 --result-format human --code-coverage"
    if target_org:
        test_cmd += f" --target-org {target_org}"

    print(f" -> Running Tests...")
    try:
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=False,
            shell=True,
            env=env
        )

        stdout_str = result.stdout.decode('utf-8', errors='replace')
        stderr_str = result.stderr.decode('utf-8', errors='replace')

        tests_passed = result.returncode == 0
        if "Test Run Failed" in stdout_str or "Fails" in stdout_str:
            tests_passed = False

        current_level = 2 if tests_passed else 1

        return {
            'level': current_level,
            'success': tests_passed,
            'stdout': stdout_str,
            'stderr': stderr_str,
            'returncode': result.returncode
        }

    except Exception as e:
        return {'level': 0, 'success': False, 'stdout': '', 'stderr': str(e), 'returncode': -1}

def save_results(
    iteration: int,
    result_dir: Path,
    files: dict,
    test_result: dict,
    response_text: str,
    diff_text: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    code_dir = result_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for filename, code in files.items():
        file_path = code_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

    if(test_result['success']):
        status = "success_"
    else:
        status = "failure_"
    with open(result_dir / f"{status}test_result.txt", 'w', encoding='utf-8') as f:
        f.write(f"Iteration {iteration}\nTimestamp: {datetime.now().isoformat()}\n")
        f.write(f"Success: {test_result['success']}\n")
        f.write("\n" + "="*60 + "\nSTDOUT:\n" + test_result['stdout'])
        f.write("\n" + "="*60 + "\nSTDERR:\n" + test_result['stderr'])

    with open(result_dir / "ai_response.txt", 'w', encoding='utf-8') as f:
        f.write(response_text)

    with open(result_dir / "diff.txt", 'w', encoding='utf-8') as f:
        f.write(diff_text or "")

def write_summary(results_dir: Path, text: str) -> None:
    with open(results_dir / f"{MODEL_SAFE}_summary_results.txt", "a", encoding="utf-8") as f:
        f.write(text)


def _usage_to_dict(usage) -> dict | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    data = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens",
                 "prompt_token_count", "candidates_token_count", "total_token_count"):
        if hasattr(usage, attr):
            data[attr] = getattr(usage, attr)
    return data or None

def format_token_usage(usage: dict | None) -> str:
    if not usage:
        return "Tokens: n/a"
    prompt = usage.get("prompt_tokens", usage.get("prompt_token_count"))
    completion = usage.get("completion_tokens", usage.get("candidates_token_count"))
    total = usage.get("total_tokens", usage.get("total_token_count"))
    parts = []
    if prompt is not None:
        parts.append(f"prompt={prompt}")
    if completion is not None:
        parts.append(f"completion={completion}")
    if total is not None:
        parts.append(f"total={total}")
    if not parts:
        return "Tokens: n/a"
    return "Tokens: " + ", ".join(parts)

def groq_generate(final_prompt: str) -> tuple[str, dict | None]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": final_prompt}]
    )
    usage = _usage_to_dict(getattr(resp, "usage", None))
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("Leere Antwort von Groq API erhalten")
    return content, usage

def gemini_generate(final_prompt: str) -> tuple[str, dict | None]:
    response = client.models.generate_content(
        model=MODEL,
        contents=final_prompt
    )

    response_text = getattr(response, "text", None)
    if not response_text and hasattr(response, "candidates"):
        parts = [p.text for c in response.candidates for p in c.content.parts if hasattr(p, "text")]
        response_text = "\n".join(parts)

    if not response_text:
        raise ValueError("Leere Antwort erhalten")

    usage = None
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is not None:
        usage = _usage_to_dict(usage_meta)

    return response_text, usage

def mistral_generate(prompt: str) -> tuple[str, dict | None]:
    res = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "content": prompt,
                "role": "user",
            },
        ],
        temperature=0.2,
        stream=False
    )
    usage = _usage_to_dict(getattr(res, "usage", None))
    content = res.choices[0].message.content
    if content is None:
        raise ValueError("Leere Antwort von Mistral API erhalten")
    return content, usage

def nvidia_generate(prompt: str) -> tuple[str, dict | None]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.6,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )
    content = ""
    for chunk in resp:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    
    if not content:
        raise ValueError("Leere Antwort von NVIDIA API erhalten")
    return content, None

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    if "status 429" in msg:
        return True
    if "rate limit" in msg:
        return True
    if '"type":"rate_limited"' in msg:
        return True
    if '"code":"1300"' in msg:
        return True
    if 'error' in msg:
        return True
    return False

def _count_lines_in_file(filepath: Path) -> int:
    try:
        if not filepath.exists():
            return 0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def _write_pmd_ruleset_file(path: Path) -> None:
    content = """<?xml version="1.0" encoding="UTF-8"?>
<ruleset name="Custom Apex Rules"
         xmlns="http://pmd.sourceforge.net/ruleset/2.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0 https://pmd.github.io/schema/ruleset_2_0_0.xsd">

    <description>Custom Cognitive Complexity threshold</description>

    <rule ref="category/apex/design.xml/CognitiveComplexity">
        <properties>
            <property name="methodReportLevel" value="1"/>
        </properties>
    </rule>

</ruleset>
"""
    path.write_text(content, encoding="utf-8")

def _materialize_subset_dir(source_root: Path, subset_root: Path, rel_paths: list[str]) -> dict[str, dict]:
    """
    Copies ONLY the given relative files from source_root into subset_root, preserving structure.
    Returns file_stats map keyed by abs path in subset dir (for LOC + complexity).
    """
    subset_root.mkdir(parents=True, exist_ok=True)
    file_stats: dict[str, dict] = {}
    for rel in rel_paths:
        rp = Path(rel)
        src = (source_root / rp).resolve()
        dst = (subset_root / rp)
        try:
            # keep safe: ensure inside source_root
            src.relative_to(source_root.resolve())
        except Exception:
            continue

        if not src.exists():
            continue
        if src.is_dir():
            continue
        if src.suffix.lower() != ".cls":
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception:
            # if copy fails, skip that file
            continue

        abs_dst = str(dst.resolve())
        file_stats[abs_dst] = {
            "relative_path": str(rp).replace("\\", "/"),
            "loc": _count_lines_in_file(dst),
            "complexity": 0,
            "is_test": "test" in rp.name.lower()
        }
    return file_stats

def build_metrics_with_pmd_subset(source_root: Path, subset_rel_paths: list[str], work_dir: Path) -> dict:
    """
    Runs PMD CognitiveComplexity ONLY on the changed files in subset_rel_paths.
    Creates a subset directory under work_dir and runs PMD on that subset directory.

    Never raises: on failure returns {"ok": False, "error": "..."}.
    """
    try:
        source_root = Path(source_root).resolve()
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        subset_root = work_dir / "subset_src"
        if subset_root.exists():
            shutil.rmtree(subset_root)

        ruleset_file = work_dir / "pmd-ruleset.xml"
        report_file = work_dir / "pmd-report.xml"

        _write_pmd_ruleset_file(ruleset_file)

        all_files_map = _materialize_subset_dir(source_root, subset_root, subset_rel_paths)

        cmd = [
            "pmd", "check",
            "--dir", str(subset_root),
            "--rulesets", str(ruleset_file),
            "--format", "xml",
            "--report-file", str(report_file)
        ]

        subprocess.run(cmd, check=False, shell=True)

        if report_file.exists():
            tree = ET.parse(str(report_file))
            root = tree.getroot()
            ns = {'pmd': 'http://pmd.sourceforge.net/report/2.0.0'}
            complexity_pattern = re.compile(r"cognitive complexity of (\d+)")

            for file_elem in root.findall('pmd:file', ns):
                report_path = file_elem.get('name')
                if not report_path:
                    continue
                abs_report_path = str(Path(report_path).resolve())

                total_complexity = 0
                for violation in file_elem.findall('pmd:violation', ns):
                    if violation.get('rule') == 'CognitiveComplexity':
                        match = complexity_pattern.search(violation.text or "")
                        if match:
                            total_complexity += int(match.group(1))

                if abs_report_path in all_files_map:
                    all_files_map[abs_report_path]['complexity'] = total_complexity

        stats = {
            "prod": {"complexity": 0, "loc": 0, "count": 0},
            "test": {"complexity": 0, "loc": 0, "count": 0},
            "total": {"complexity": 0, "loc": 0, "count": 0},
        }

        for data in all_files_map.values():
            category = "test" if data.get("is_test") else "prod"
            stats[category]["complexity"] += int(data.get("complexity", 0) or 0)
            stats[category]["loc"] += int(data.get("loc", 0) or 0)
            stats[category]["count"] += 1

        stats["total"]["complexity"] = stats["prod"]["complexity"] + stats["test"]["complexity"]
        stats["total"]["loc"] = stats["prod"]["loc"] + stats["test"]["loc"]
        stats["total"]["count"] = stats["prod"]["count"] + stats["test"]["count"]

        files_list = [v["relative_path"] for v in sorted(all_files_map.values(), key=lambda x: x["relative_path"])]

        return {
            "ok": True,
            "timestamp": datetime.now().isoformat(),
            "scope": {
                "mode": "subset_changed_files_only",
                "source_root": str(source_root),
                "subset_root": str(subset_root),
                "files": files_list,
            },
            "pmd": {
                "ruleset_file": str(ruleset_file),
                "report_file": str(report_file) if report_file.exists() else None,
            },
            "summary": stats
        }

    except Exception as e:
        return {
            "ok": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }

def save_metrics(result_dir: Path, metrics: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _get_refactoring_type_from_path(refactoring_path: str) -> str:
    """
    Extracts refactoring type from paths like 'refactoring/coc_reduktion'.
    """
    name = Path(refactoring_path).name.strip().lower()
    return name

def _strip_apex_comments(code: str) -> str:
    """
    Removes // line comments and /* */ block comments (best-effort).
    """
    if not code:
        return ""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    return code

def _read_project_file_text(project_root: Path, rel_path: str) -> str:
    p = (project_root / Path(rel_path)).resolve()
    try:
        p.relative_to(project_root.resolve())
    except Exception:
        return ""
    return _read_text_best_effort(p)

def _iter_apex_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'tests', 'test', 'pathlib2.egg-info', '.git'}]
        for fn in filenames:
            if fn.lower().endswith((".cls", ".trigger")):
                files.append(Path(root) / fn)
    return files

def _scan_project_for_regex(project_root: Path, pattern: str) -> list[str]:
    """
    Returns list of relative file paths where pattern matches (comments stripped).
    """
    rx = re.compile(pattern)
    hits: list[str] = []
    for fp in _iter_apex_files(project_root):
        rel = str(fp.relative_to(project_root)).replace("\\", "/")
        content = _strip_apex_comments(_read_text_best_effort(fp))
        if rx.search(content):
            hits.append(rel)
    return hits

def _extract_method_body_apex(code: str, method_name: str) -> str | None:
    """
    Best-effort extraction: finds first occurrence of '<name>(' and returns brace-matched body.
    """
    if not code or not method_name:
        return None
    code_nc = _strip_apex_comments(code)
    # Match typical Apex signatures: modifiers + return type + name + '('
    sig_rx = re.compile(r"\b" + re.escape(method_name) + r"\s*\(", re.MULTILINE)
    m = sig_rx.search(code_nc)
    if not m:
        return None

    # Find first '{' after signature
    start = code_nc.find("{", m.end())
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(code_nc)):
        ch = code_nc[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code_nc[start:i+1]
    return None

def _max_if_nesting_apex(code: str) -> int:
    """
    Heuristic nesting score: counts max number of active 'if' blocks based on brace depth.
    """
    code_nc = _strip_apex_comments(code)
    if not code_nc:
        return 0

    # Track brace depth and count of 'if' entering at each depth
    depth = 0
    max_score = 0
    if_stack: list[int] = []

    token_rx = re.compile(r"\bif\s*\(|\{|\}", re.MULTILINE)
    for m in token_rx.finditer(code_nc):
        tok = m.group(0)
        if tok == "{":
            depth += 1
        elif tok == "}":
            # Pop any ifs that were recorded at deeper depths
            while if_stack and if_stack[-1] >= depth:
                if_stack.pop()
            depth = max(depth - 1, 0)
        else:
            # if(
            if_stack.append(depth)
            max_score = max(max_score, len(if_stack))
    return max_score


def _count_guard_clauses_apex(code: str) -> int:
    """
    Heuristic count of 'guard clauses' in Apex: conditionals that lead to an early exit
    (return/continue/break/throw) very shortly after the condition.

    Counts patterns like:
      - if (cond) return ...;
      - if (cond) { return ...; }
      - if (cond) continue;
      - if (cond) throw ...;

    Best-effort only (comments removed before scanning).
    """
    code_nc = _strip_apex_comments(code)
    if not code_nc:
        return 0

    # Normalize whitespace to make the regex more stable.
    s = re.sub(r"\s+", " ", code_nc)

    # 1) Single-line guard: if (...) return|continue|break|throw
    rx_inline = re.compile(r"\bif\s*\([^\)]*\)\s*(?:\{|)\s*(return|continue|break|throw)\b")
    count = len(rx_inline.findall(s))

    # 2) Braced blocks where early-exit occurs shortly after the if-condition.
    #    We allow a small window of non-brace tokens before the exit keyword.
    rx_block = re.compile(
        r"\bif\s*\([^\)]*\)\s*\{[^\}]{0,200}?\b(return|continue|break|throw)\b"
    )
    count = max(count, 0) + len(rx_block.findall(s))

    return count

def _parse_prompt_targets(prompt_text: str, ref_type: str) -> dict:
    """
    Extracts task-specific targets from the prompt text using regex.
    Returns a dict with keys depending on ref_type.
    """
    t = prompt_text or ""
    ref_type = (ref_type or "").lower()

    if ref_type == "rename":
        m = re.search(
            r"Rename the method\s+`([^`]+)`\s+in the file:\s+`([^`]+)`\s+to\s+`([^`]+)`",
            t,
            flags=re.IGNORECASE
        )
        if m:
            return {"old_method": m.group(1), "file": m.group(2), "new_method": m.group(3)}

    if ref_type == "inline_variable":
        m = re.search(
            r"Inline the temporary variable\s+`([^`]+)`\s+in the method\s+`?([A-Za-z_]\w*)`?\s+in the file\s+`?([^`\s]+)`?\s*\.?",
            t,
            flags=re.IGNORECASE
        )
        if m:
            return {"var": m.group(1), "method": m.group(2), "file": m.group(3)}


    if ref_type == "getter_setter":
        m = re.search(
            r"Encapsulate only the attribute\s+`([^`]+)`\s+within the class\s+`([^`]+)`\s+in the file\s+`([^`]+)`",
            t,
            flags=re.IGNORECASE
        )
        if m:
            return {"attr": m.group(1), "class": m.group(2), "file": m.group(3)}

    if ref_type == "strategy_pattern":
        m = re.search(
            r"Refactor the method\s+`?([A-Za-z_]\w*)`?\s+in the file\s+`([^`]+)`\s+to use the Strategy pattern",
            t,
            flags=re.IGNORECASE
        )
        if m:
            return {"method": m.group(1), "file": m.group(2)}

    return {}

def build_refactoring_check(
    refactoring_path: str,
    prompt_text: str,
    backup_dir: Path,
    project_dir: Path,
    changed_rel_paths: list[str],
    metrics: dict | None = None,
) -> dict:
    """
    Returns a dict for metrics.json: { "type": ..., "ok": bool, "checks": {...} }

    Scope:
    - Runs only the checks relevant to refactoring type derived from refactoring_path.
    - Uses regex-parsed targets from prompt_text wherever possible.
    """
    ref_type = _get_refactoring_type_from_path(refactoring_path)
    targets = _parse_prompt_targets(prompt_text, ref_type)

    checks: dict[str, dict] = {}
    overall_ok = True

    # ---- helper for recording results ----
    def _record(name: str, passed: bool, details: dict) -> None:
        nonlocal overall_ok
        checks[name] = {"passed": bool(passed), **(details or {})}
        if not passed:
            overall_ok = False

    # ---- rename ----
    if ref_type == "rename":
        old_m = targets.get("old_method")
        new_m = targets.get("new_method")
        file_hint = targets.get("file")

        if not old_m or not new_m:
            _record("rename", False, {"reason": "targets_not_parsed", "targets": targets})
        else:
            # 1) Resolve file hint to an actual relative path in the repo
            resolved_file = _resolve_file_hint(project_dir, file_hint, changed_rel_paths) if file_hint else None

            # 2) Scan project usage (calls + possible decl). comments already stripped in helper.
            old_hits = _scan_files_for_regex(project_dir, changed_rel_paths, r"\b" + re.escape(old_m) + r"\s*\(")
            new_hits = _scan_files_for_regex(project_dir, changed_rel_paths, r"\b" + re.escape(new_m) + r"\s*\(")

            # if I want to scan the project, this suits only if there is no other method with the same name but in another class 
            # old_hits = _scan_project_for_regex(project_dir, r"\b" + re.escape(old_m) + r"\s*\(")
            # new_hits = _scan_project_for_regex(project_dir, r"\b" + re.escape(new_m) + r"\s*\(")


            # 3) Verify declaration in resolved file (most important)
            decl_old = None
            decl_new = None
            hinted_text = ""
            if resolved_file:
                hinted_text = _strip_apex_comments(_read_project_file_text(project_dir, resolved_file))

                # Apex return types can contain dots (ApexPages.Component) + generics + arrays etc.
                type_rx = r"[\w.<>\[\],\s]+"

                sig_old = re.compile(
                    r"\b(?:public|private|protected|global)\s+"
                    r"(?:static\s+)?"
                    + type_rx + r"\s+"
                    + re.escape(old_m) + r"\s*\(",
                    flags=re.IGNORECASE
                )
                sig_new = re.compile(
                    r"\b(?:public|private|protected|global)\s+"
                    r"(?:static\s+)?"
                    + type_rx + r"\s+"
                    + re.escape(new_m) + r"\s*\(",
                    flags=re.IGNORECASE
                )

                decl_old = sig_old.search(hinted_text) is not None
                decl_new = sig_new.search(hinted_text) is not None

            # 4) Decide pass/fail (strict but correct)
            reasons: list[str] = []
            passed = True

            if not resolved_file:
                passed = False
                reasons.append("file_hint_not_resolved")

            # Declaration must be renamed in the target file
            if resolved_file:
                if decl_old is True:
                    passed = False
                    reasons.append("old_method_declaration_still_present_in_file")
                if decl_new is not True:
                    passed = False
                    reasons.append("new_method_declaration_not_found_in_file")

            # Old name must not be referenced anywhere anymore (project-wide)
            if len(old_hits) != 0:
                passed = False
                reasons.append("old_method_still_referenced")

            _record(
                "rename",
                passed,
                {
                    "old_method": old_m,
                    "new_method": new_m,
                    "file_hint": file_hint,
                    "resolved_file": resolved_file,
                    "decl_old_in_file": decl_old,
                    "decl_new_in_file": decl_new,
                    "old_method_hits": old_hits[:25],
                    "new_method_hits": new_hits[:25],
                    "reasons": reasons,
                },
            )



    # ---- inline_variable ----
    if ref_type == "inline_variable":
        var_name = targets.get("var")
        method_name = targets.get("method")
        file_hint = targets.get("file")

        if not var_name or not method_name or not file_hint:
            _record("inline_variable", False, {"reason": "targets_not_parsed", "targets": targets})
        else:
            resolved = _resolve_file_hint(project_dir, file_hint, changed_rel_paths)
            if not resolved:
                _record("inline_variable", False, {"reason": "file_not_found_in_project", "file_hint": file_hint})
            else:
                after_text = _read_project_file_text(project_dir, resolved)
                before_text = _read_text_best_effort(backup_dir / Path(resolved))

                after_nc = _strip_apex_comments(after_text)
                before_nc = _strip_apex_comments(before_text)

                after_body = _extract_method_body_apex(after_nc, method_name)
                before_body = _extract_method_body_apex(before_nc, method_name)

                decl_rx = re.compile(r"\b\w+\s+" + re.escape(var_name) + r"\s*=")
                return_rx = re.compile(r"\breturn\s+" + re.escape(var_name) + r"\s*;")

                passed = True
                reasons: list[str] = []

                if before_body is None or after_body is None:
                    passed = False
                    reasons.append("method_not_found")

                if after_body is not None:
                    if decl_rx.search(after_body):
                        passed = False
                        reasons.append("var_declaration_still_present")
                    if return_rx.search(after_body):
                        passed = False
                        reasons.append("return_var_still_present")

                had_decl_before = bool(before_body and decl_rx.search(before_body))
                use_rx = re.compile(r"\b" + re.escape(var_name) + r"\b")
                if after_body is not None and use_rx.search(after_body):
                    passed = False
                    reasons.append("var_usage_still_present")


                _record(
                    "inline_variable",
                    passed,
                    {
                        "file": resolved,  # <-- wichtig
                        "file_hint": file_hint,
                        "method": method_name,
                        "var": var_name,
                        "had_declaration_before": had_decl_before,
                        "reasons": reasons,
                    },
                )


    # ---- getter_setter ----
    if ref_type == "getter_setter":
        attr = targets.get("attr")
        cls = targets.get("class")
        file_hint = targets.get("file")

        if not attr or not cls or not file_hint:
            _record("getter_setter", False, {"reason": "targets_not_parsed", "targets": targets})
        else:
            resolved = _resolve_file_hint(project_dir, file_hint, changed_rel_paths)
            if not resolved:
                _record("getter_setter", False, {"reason": "file_not_found_in_project", "file_hint": file_hint})
            else:
                after_text = _strip_apex_comments(_read_project_file_text(project_dir, resolved))

                cap = attr[:1].upper() + attr[1:]
                get_name = f"get{cap}"
                set_name = f"set{cap}"

                # getter signature examples:
                # public String getName() { ... }
                # global static Integer getFoo() { ... }
                getter_rx = re.compile(
                    r"\b(?:public|private|protected|global)\s+"
                    r"(?:static\s+)?"
                    r"[\w<>\[\]]+\s+"
                    + re.escape(get_name) +
                    r"\s*\(\s*\)",
                    flags=re.IGNORECASE
                )
                # setter signature examples:
                # public void setName(String v) { ... }
                setter_rx = re.compile(
                    r"\b(?:public|private|protected|global)\s+"
                    r"(?:static\s+)?"
                    r"void\s+"
                    + re.escape(set_name) +
                    r"\s*\(\s*[\w<>\[\]]+\s+\w+\s*\)",
                    flags=re.IGNORECASE
                )

                has_get = getter_rx.search(after_text) is not None
                has_set = setter_rx.search(after_text) is not None

                # Public field patterns to avoid:
                # public String name;
                # public static final Integer name = 1;
                public_field_rx = re.compile(
                    r"\bpublic\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+"
                    + re.escape(attr) +
                    r"\b",
                    flags=re.IGNORECASE
                )
                public_field = public_field_rx.search(after_text) is not None

                # Optional: encourage private field presence (common encapsulation)
                private_field_rx = re.compile(
                    r"\bprivate\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+"
                    + re.escape(attr) +
                    r"\b",
                    flags=re.IGNORECASE
                )
                private_field_present = private_field_rx.search(after_text) is not None

                # Soft-signal only (do NOT hard fail):
                # `.name` can appear legitimately; we just record hits for analysis.
                direct_access_hits = _scan_project_for_regex(project_dir, r"\." + re.escape(attr) + r"\b")

                # Pass criteria: getter+setter exist AND no public field remains.
                passed = has_get and has_set and (not public_field)

                reasons = []
                if not has_get:
                    reasons.append("getter_not_found")
                if not has_set:
                    reasons.append("setter_not_found")
                if public_field:
                    reasons.append("public_field_still_present")

                _record(
                    "getter_setter",
                    passed,
                    {
                        "file": resolved,
                        "file_hint": file_hint,
                        "class": cls,
                        "attr": attr,
                        "expected_getter": get_name,
                        "expected_setter": set_name,
                        "has_getter": has_get,
                        "has_setter": has_set,
                        "public_field_still_present": public_field,
                        "private_field_present": private_field_present,
                        "direct_access_hits_count": len(direct_access_hits),
                        "direct_access_hits_sample": direct_access_hits[:25],
                        "reasons": reasons,
                        "note": "getter_setter_requires_methods_and_no_public_field; dot-access_is_soft_signal",
                    },
                )



    # ---- guard_clauses ----
    if ref_type == "guard_clauses":
        # Guard clauses are about EARLY EXITS (return/continue/break/throw) to reduce nesting.
        # Nesting reduction is helpful but not required in all valid refactors (e.g., single-level `if` -> `if(!cond) return;`).
        improved_files: list[str] = []
        compared: list[dict] = []

        total_guard_delta = 0
        total_nesting_delta = 0

        for rel in changed_rel_paths:
            if not rel.lower().endswith(".cls"):
                continue

            before_src = _strip_apex_comments(_read_text_best_effort(backup_dir / Path(rel)))
            after_src = _strip_apex_comments(_read_project_file_text(project_dir, rel))

            b_nest = _max_if_nesting_apex(before_src)
            a_nest = _max_if_nesting_apex(after_src)

            b_guard = _count_guard_clauses_apex(before_src)
            a_guard = _count_guard_clauses_apex(after_src)

            nest_delta = a_nest - b_nest
            guard_delta = a_guard - b_guard

            total_guard_delta += guard_delta
            total_nesting_delta += nest_delta

            compared.append(
                {
                    "file": rel,
                    "before_max_if_nesting": b_nest,
                    "after_max_if_nesting": a_nest,
                    "before_guard_clauses": b_guard,
                    "after_guard_clauses": a_guard,
                    "delta_max_if_nesting": nest_delta,
                    "delta_guard_clauses": guard_delta,
                }
            )

            # Consider a file "improved" if:
            # - nesting decreases, OR
            # - guard clauses (early exits) increase without increasing nesting a lot.
            if (a_nest < b_nest) or (guard_delta > 0 and a_nest <= b_nest + 1):
                improved_files.append(rel)

        # Pass condition:
        # - at least one changed file shows improvement per above heuristic.
        # This avoids false negatives when guard clauses increase but nesting stays equal.
        passed = len(improved_files) > 0

        _record(
            "guard_clauses",
            passed,
            {
                "improved_files": improved_files[:25],
                "comparisons": compared[:25],
                "totals": {
                    "delta_guard_clauses": total_guard_delta,
                    "delta_max_if_nesting": total_nesting_delta,
                },
                "note": "heuristic_early_exit_and_nesting",
            },
        )

    # ---- strategy_pattern ----
    if ref_type == "strategy_pattern":
        method_name = targets.get("method")
        file_hint = targets.get("file")

        if not method_name or not file_hint:
            _record("strategy_pattern", False, {"reason": "targets_not_parsed", "targets": targets})
        else:
            resolved = _resolve_file_hint(project_dir, file_hint, changed_rel_paths)
            if not resolved:
                _record("strategy_pattern", False, {"reason": "file_not_found_in_project", "file_hint": file_hint})
            else:
                before_text = _strip_apex_comments(_read_text_best_effort(backup_dir / Path(resolved)))
                after_text = _strip_apex_comments(_read_project_file_text(project_dir, resolved))

                before_body = _extract_method_body_apex(before_text, method_name) or ""
                after_body = _extract_method_body_apex(after_text, method_name) or ""

                before_else_if = len(re.findall(r"\belse\s+if\b", before_body))
                after_else_if = len(re.findall(r"\belse\s+if\b", after_body))

                # --- Strategy signals in FILE (interface + implementations) ---
                # Prefer explicit interface name if present (IdFieldStrategy etc.)
                iface_names = set(re.findall(r"\binterface\s+([A-Za-z_]\w*)\b", after_text))
                # Fall back: any interface containing 'Strategy' in the name
                iface_names |= {n for n in iface_names if "strategy" in n.lower()}

                has_any_interface = len(iface_names) > 0
                has_strategy_named_interface = any("strategy" in n.lower() for n in iface_names)

                implements_hits = []
                for iface in iface_names:
                    if re.search(r"\bclass\s+[A-Za-z_]\w*\s+implements\s+" + re.escape(iface) + r"\b", after_text):
                        implements_hits.append(iface)
                has_implementation = len(implements_hits) > 0

                # --- Strategy usage in METHOD (delegation) ---
                # Look for var.methodName(...) call
                # (we don't know the strategy var name, so generic: "<ident>.<ident>(")
                has_delegation_call = re.search(r"\b[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*\s*\(", after_body) is not None

                # --- Selection mechanism (either conditional new, or map dispatch) ---
                has_map = re.search(r"\bMap\s*<", after_text) is not None
                has_contains = re.search(r"\bcontainsKey\s*\(", after_text) is not None
                uses_map_dispatch = has_map and has_contains

                # conditional selection: assigns "new Something()" into some variable inside the method
                # (common in valid strategy refactors)
                has_new_assignment = re.search(r"=\s*new\s+[A-Za-z_]\w*\s*\(", after_body) is not None

                # Strategy variant label (helps later analysis)
                if uses_map_dispatch:
                    variant = "map_dispatch"
                elif has_new_assignment:
                    variant = "conditional_selection"
                else:
                    variant = "unknown_selection"

                # --- Pass criteria ---
                # Minimum: interface + implementation + delegation
                passed = has_any_interface and has_implementation and has_delegation_call

                reasons = []
                if not has_any_interface:
                    reasons.append("no_interface_found")
                if not has_implementation:
                    reasons.append("no_implements_found")
                if not has_delegation_call:
                    reasons.append("no_delegation_call_in_method")
                if variant == "unknown_selection":
                    # don't hard-fail for this; just record it (some strategies are injected)
                    reasons.append("no_obvious_selection_mechanism")

                _record(
                    "strategy_pattern",
                    passed,
                    {
                        "file": resolved,
                        "file_hint": file_hint,
                        "method": method_name,
                        "else_if_before": before_else_if,
                        "else_if_after": after_else_if,
                        "has_map": has_map,
                        "has_containsKey": has_contains,
                        "variant": variant,
                        "interfaces_found": sorted(list(iface_names))[:10],
                        "implements_interfaces": implements_hits[:10],
                        "has_delegation_call": has_delegation_call,
                        "has_new_assignment_in_method": has_new_assignment,
                        "reasons": reasons,
                        "note": "heuristic_strategy_interface_impl_delegation",
                    },
                )


    # ---- coc_reduktion ----
    if ref_type == "coc_reduktion":
        # Use PMD delta if available, else fall back to nesting heuristic.
        delta_total = None
        if metrics and isinstance(metrics, dict):
            d = metrics.get("pmd_delta", {}).get("total", {}).get("complexity")
            if d is not None:
                try:
                    delta_total = int(d)
                except Exception:
                    delta_total = None

        if delta_total is not None:
            passed = delta_total < 0
            _record(
                "coc_reduktion",
                passed,
                {"pmd_delta_total_complexity": delta_total, "rule": "delta<=0"},
            )
        else:
            improved_files: list[str] = []
            compared: list[dict] = []
            for rel in changed_rel_paths:
                if not rel.lower().endswith(".cls"):
                    continue
                before = _strip_apex_comments(_read_text_best_effort(backup_dir / Path(rel)))
                after = _strip_apex_comments(_read_project_file_text(project_dir, rel))
                b = _max_if_nesting_apex(before)
                a = _max_if_nesting_apex(after)
                compared.append({"file": rel, "before_max_if_nesting": b, "after_max_if_nesting": a})
                if a <= b:
                    improved_files.append(rel)
            passed = len(improved_files) > 0
            _record(
                "coc_reduktion",
                passed,
                {
                    "improved_files": improved_files[:25],
                    "comparisons": compared[:25],
                    "note": "fallback_heuristic_if_nesting_non_increase",
                },
            )

    # If unknown type: mark as n/a but ok
    if ref_type not in {"rename", "inline_variable", "getter_setter", "guard_clauses", "strategy_pattern", "coc_reduktion"}:
        _record("refactoring_type", True, {"note": "unknown_refactoring_type_no_checks_run", "type": ref_type})

    return {
        "type": ref_type,
        "targets": targets,
        "ok": bool(overall_ok),
        "checks": checks,
    }

def format_refactoring_check_summary(ref_check: dict | None) -> str:
    """
    Returns a short one-liner: "REF:ok" or "REF:fail(<check1>,<check2>)"
    """
    if not ref_check:
        return "REF:n/a"
    if ref_check.get("ok"):
        return "REF:ok"
    checks = ref_check.get("checks", {}) or {}
    failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("passed")]
    if not failed:
        return "REF:fail"
    return "REF:fail(" + ",".join(failed[:3]) + ")"

def main():
    PROJECT_DIR = Path(args.project_path)

    if args.all_refactorings:
        selected_refactorings = REFACTORINGS
    else:
        selected_refactorings = [args.refactoring]

    print(f"{'='*60}\nStarte Refactoring-Experiment\n{'='*60}\n")

    backup_dir = Path("backup_original")
    backup_project(PROJECT_DIR, backup_dir)

    for ref_name in selected_refactorings:
        REFACTORING = f"{REFACTORING_BASE_DIR}/{ref_name}"

        PROMPT_TEMPLATE = Path(f"{REFACTORING}.txt").read_text(encoding="utf-8")
        RESULTS_DIR = Path(REFACTORING + RESULT_PATH + MODEL_SAFE)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        YOUR_PROMPT = PROMPT_TEMPLATE

        print(f"{'='*60}\nRefactoring: {ref_name}\n{'='*60}\n")

        project_structure = get_project_structure(PROJECT_DIR)
        code_block = get_all_apex_files(PROJECT_DIR)
        final_prompt = f"{YOUR_PROMPT}\n\nStruktur:\n{project_structure}\n\nCode:\n{code_block}"

        with open(RESULTS_DIR / "full_prompt.txt", "w", encoding="utf-8") as f:
            f.write(final_prompt)

        successful_iterations = 0

        i = 1
        while i <= ITERATIONS:
            print(f"\nITERATION {i}/{ITERATIONS}")
            restore_project(backup_dir, PROJECT_DIR)

            iteration_dir = RESULTS_DIR / f"iteration_{i:02d}"

            try:
                usage = None
                if LLM_API_KEY == MISTRAL_API_KEY:
                    response_text, usage = mistral_generate(final_prompt)
                elif LLM_API_KEY == GEMINI_API_KEY:
                    response_text, usage = gemini_generate(final_prompt)
                elif LLM_API_KEY == GROQ_API_KEY:
                    response_text, usage = groq_generate(final_prompt)
                elif LLM_API_KEY == NVIDIA_API_KEY:
                    response_text, usage = nvidia_generate(final_prompt)
                else:
                    raise RuntimeError("Kein gültiger LLM_API_KEY gesetzt")

                files = parse_ai_response(response_text)
                if not files:
                    i += 1
                    continue

                changed_rel_paths = sorted({str(Path(p)) for p in files.keys()})
                changed_rel_paths = [p for p in changed_rel_paths if p.lower().endswith(".cls")]

                metrics_before = build_metrics_with_pmd_subset(
                    PROJECT_DIR,
                    changed_rel_paths,
                    iteration_dir / "pmd_before"
                )

                apply_changes(PROJECT_DIR, files)

                metrics_after = build_metrics_with_pmd_subset(
                    PROJECT_DIR,
                    changed_rel_paths,
                    iteration_dir / "pmd_after"
                )

                has_diff, diff_text = build_diff_between_backup_and_refactored(
                    backup_dir=backup_dir,
                    project_src=PROJECT_DIR,
                    snapshot_files=files,
                )
                diff_status = "passed" if has_diff else "failed"

                test_result = run_apex_tests()
                test_status = "passed" if test_result.get("success") else "failed"

                token_info = format_token_usage(usage)
                pmd_summary = format_pmd_metrics_summary(metrics_before, metrics_after)

                save_results(i, iteration_dir, files, test_result, response_text, diff_text)

                metrics = {
                    "iteration": i,
                    "timestamp": datetime.now().isoformat(),
                    "refactoring": REFACTORING,
                    "changed_files_scope": changed_rel_paths,
                    "test": {"status": test_status, "success": bool(test_result.get("success")), "level": test_result.get("level")},
                    "diff": {"status": diff_status, "has_diff": bool(has_diff)},
                    "tokens": token_info,
                    "pmd_before": metrics_before,
                    "pmd_after": metrics_after,
                }

                if metrics_before.get("ok") and metrics_after.get("ok"):
                    b = metrics_before.get("summary", {})
                    a = metrics_after.get("summary", {})

                    def _delta(cat: str, key: str) -> int:
                        return int((a.get(cat, {}).get(key, 0) or 0)) - int((b.get(cat, {}).get(key, 0) or 0))

                    metrics["pmd_delta"] = {
                        "prod": {"complexity": _delta("prod", "complexity"), "loc": _delta("prod", "loc"), "count": _delta("prod", "count")},
                        "test": {"complexity": _delta("test", "complexity"), "loc": _delta("test", "loc"), "count": _delta("test", "count")},
                        "total": {"complexity": _delta("total", "complexity"), "loc": _delta("total", "loc"), "count": _delta("total", "count")},
                    }

                ref_check = build_refactoring_check(
                    refactoring_path=REFACTORING,
                    prompt_text=YOUR_PROMPT,
                    backup_dir=backup_dir,
                    project_dir=PROJECT_DIR,
                    changed_rel_paths=changed_rel_paths,
                    metrics=metrics,
                )
                metrics["refactoring_check"] = ref_check

                ref_summary = format_refactoring_check_summary(ref_check)
                ref_ok = bool(ref_check.get("ok")) if isinstance(ref_check, dict) else False

                iteration_status = "passed" if (test_result.get("success") and has_diff and ref_ok) else "failed"

                if iteration_status == "passed":
                    successful_iterations += 1

                line = f"iteration {i} {iteration_status} test {test_status} diff {diff_status} {token_info} {pmd_summary} {ref_summary}\n"
                write_summary(RESULTS_DIR, line)
                print(line.strip())

                save_metrics(iteration_dir, metrics)

                i += 1

            except Exception as e:
                if _is_rate_limit_error(e):
                    print(f"Fehler: {e}")
                    print("Rate limit erhalten. Warte 60 Sekunden und wiederhole die Iteration.")
                    time.sleep(60)
                    continue
                print(f"Fehler: {e}")
                i += 1

        success_rate = (successful_iterations / ITERATIONS * 100.0) if ITERATIONS else 0.0
        final_line = f"Fertig. Erfolgsrate: {success_rate:.1f}% ({successful_iterations}/{ITERATIONS})\n"
        write_summary(RESULTS_DIR, final_line)
        print(final_line.strip())

    restore_project(backup_dir, PROJECT_DIR)


if __name__ == "__main__":
    main()
