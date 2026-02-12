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

REFACTORING = 'refactoring/coc_reduktion'
PATH = 'force-di'
ITERATIONS = 10
GEMINI3 = 'gemini-3-pro-preview'
GEMINI2 = 'gemini-2.5-flash'
LLAMA = 'llama-3.3-70b-versatile'
MISTRAL = 'mistral-large-2512'
CODESTRAL = 'codestral-2501'
MODEL_OLLAMA = 'devstral-2_123b-cloud'
MODEL_GROQ = LLAMA
MODEL_GEMINI = GEMINI3
MODEL_MISTRAL = CODESTRAL
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
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

parser = argparse.ArgumentParser(description="Projektpfad angeben")
parser.add_argument("--project-path", type=str, default=PATH, help="Pfad des Projekts")
args = parser.parse_args()

PROJECT_DIR = Path(args.project_path)
PROMPT_TEMPLATE = Path(f"{REFACTORING}.txt").read_text(encoding='utf-8')
RESULTS_DIR = Path(REFACTORING + "_results_" + MODEL)
RESULTS_DIR.mkdir(exist_ok=True)


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

def write_summary(text: str) -> None:
    with open(RESULTS_DIR / f"{MODEL}_summary_results.txt", "a", encoding="utf-8") as f:
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
        content=final_prompt
    )
    usage = _usage_to_dict(getattr(resp, "usage", None))
    return resp.choices[0].message.content, usage

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
    return res.choices[0].message.content, usage

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    if "Status 429" in msg:
        return True
    if "rate limit" in msg.lower():
        return True
    if '"type":"rate_limited"' in msg:
        return True
    if '"code":"1300"' in msg:
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

def main():
    YOUR_PROMPT = PROMPT_TEMPLATE
    print(f"{'='*60}\nStarte Refactoring-Experiment\n{'='*60}\n")

    backup_dir = Path("backup_original")
    backup_project(PROJECT_DIR, backup_dir)

    project_structure = get_project_structure(PROJECT_DIR)
    code_block = get_all_apex_files(PROJECT_DIR)

    final_prompt = f"{YOUR_PROMPT}\n\nStruktur:\n{project_structure}\n\nCode:\n{code_block}"
    successful_iterations = 0

    with open(RESULTS_DIR / "full_prompt.txt", "w", encoding="utf-8") as f:
        f.write(final_prompt)

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

            files = parse_ai_response(response_text)
            if not files:
                i += 1
                continue

            # Only changed files that are real Apex classes
            changed_rel_paths = sorted({str(Path(p)) for p in files.keys()})
            changed_rel_paths = [p for p in changed_rel_paths if p.lower().endswith(".cls")]

            # PMD metrics BEFORE changes: subset only (based on original restored project)
            metrics_before = build_metrics_with_pmd_subset(PROJECT_DIR, changed_rel_paths, iteration_dir / "pmd_before")

            apply_changes(PROJECT_DIR, files)

            # PMD metrics AFTER changes: subset only (same file set, new content)
            metrics_after = build_metrics_with_pmd_subset(PROJECT_DIR, changed_rel_paths, iteration_dir / "pmd_after")

            has_diff, diff_text = build_diff_between_backup_and_refactored(
                backup_dir=backup_dir,
                project_src=PROJECT_DIR,
                snapshot_files=files,
            )
            diff_status = "passed" if has_diff else "failed"

            test_result = run_apex_tests()
            test_status = "passed" if test_result['success'] else "failed"

            iteration_status = "passed" if (test_result['success'] and has_diff) else "failed"
            token_info = format_token_usage(usage)

            if iteration_status == "passed":
                successful_iterations += 1

            pmd_summary = format_pmd_metrics_summary(metrics_before, metrics_after)
            write_summary(f"iteration {i} {iteration_status} test {test_status} diff {diff_status} {token_info} {pmd_summary}\n")


            if test_result['success']:
                print(" Tests bestanden.")
            else:
                print(" Tests fehlgeschlagen.")

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

    print(f"\nFertig. Erfolgsrate: {successful_iterations/ITERATIONS*100:.1f}%")
    restore_project(backup_dir, PROJECT_DIR)

if __name__ == "__main__":
    main()
