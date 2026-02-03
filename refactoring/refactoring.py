import os
from pyexpat import model
import re
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from unittest import result

REFACTORING = 'refactoring/coc_reduktion'
PATH = 'force-di'
ITERATIONS = 1
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

def get_project_structure(project_dir: Path) -> str:
    """Erstellt eine Übersicht der Projektstruktur."""
    structure = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'tests', 'pathlib2.egg-info'}]
        level = root.replace(str(project_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        structure.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.cls') or file.endswith('.trigger'):
                structure.append(f'{subindent}{file}')
    return '\n'.join(structure)

def get_all_apex_files(project_dir: Path) -> str:
    """Liest alle Apex-Dateien ein und liefert einen großen Textblock."""
    code_block = ""
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'tests', 'pathlib2.egg-info'}]
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
    """Parst die AI-Antwort und extrahiert Dateinamen und Code."""
    files = {}
    pattern = r"File\s+`([^`]+)`:\s*```apex\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    for filename, code in matches:
        files[filename] = code.strip()
    return files

def backup_project(project_dir: Path, backup_dir: Path) -> None:
    """Erstellt ein Backup des Projekts."""
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(
        project_dir, backup_dir, 
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'test', 'tests', 'pathlib2.egg-info')
    )

def restore_project(backup_dir: Path, project_dir: Path) -> None:
    """Stellt das Projekt aus dem Backup wieder her"""
    backup_dir = Path(backup_dir).resolve()
    project_dir = Path(project_dir).resolve()

    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup-Verzeichnis nicht gefunden: {backup_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(backup_dir, project_dir, dirs_exist_ok=True)

def apply_changes(project_dir: Path | str, files: dict[str, str]) -> None:
    """Wendet die Änderungen auf die Dateien an, ignoriert jedoch Dateien im 'tests'-Ordner."""
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
    """
    Returns a dictionary with 'level':
    Level 0: Deployment Failed (Compilation Error)
    Level 1: Deployment Success, Tests Failed
    Level 2: Deployment Success, Tests Passed
    """
    
    # Umgebungsvariablen für diesen Prozess anpassen
    # Wir sagen der SF CLI: "Wir sind ein Roboter (CI), bitte keine bunten Emojis und Ladebalken."
    env = os.environ.copy()
    env["CI"] = "true"
    env["SF_NO_COLOR"] = "true"
    env["SF_DISABLE_AUTOUPDATE"] = "true"

    # --- LEVEL 0 CHECK: DEPLOYMENT ---
    deploy_cmd = "sf project deploy start --ignore-conflicts"
    if target_org:
        deploy_cmd += f" --target-org {target_org}"
        
    print(f" -> Deploying Code...")
    
    # WICHTIG: text=False (Binary Mode). Python fasst die Daten nicht an -> Kein Absturz.
    deploy_res = subprocess.run(
        deploy_cmd, 
        capture_output=True, 
        text=False,          # <--- Binary Mode!
        shell=True, 
        env=env              # <--- Saubere Umgebung
    )
    
    # Manuelles Decodieren (Sicher)
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

    # --- LEVEL 1 & 2 CHECK: TESTS ---
    test_cmd = "sf apex run test --wait 10 --result-format human --code-coverage"
    if target_org:
        test_cmd += f" --target-org {target_org}"
    
    print(f" -> Running Tests...")
    try:
        # Auch hier: text=False
        result = subprocess.run(
            test_cmd, 
            capture_output=True, 
            text=False,      # <--- Binary Mode!
            shell=True,
            env=env
        )
        
        # Manuelles Decodieren
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

def save_results(iteration: int, result_dir: Path, files: dict, test_result: dict, response_text: str) -> None:
    """Speichert die Ergebnisse einer Iteration."""
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
    """Fragt Gemini (Text Completions) an und gibt den Text-Content zurück."""
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

    for i in range(1, ITERATIONS   +1):
        print(f"\nITERATION {i}/{ITERATIONS}")
        restore_project(backup_dir, PROJECT_DIR)

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
                continue

            apply_changes(PROJECT_DIR, files)
            test_result = run_apex_tests()

            token_info = format_token_usage(usage)

            if test_result['success']:
                successful_iterations += 1
                write_summary(f"\nIteration {i} erfolgreich. {token_info}")
                print(" Tests bestanden.")
            else:
                write_summary(f"\nIteration {i} fehlgeschlagen. {token_info}")
                print(" Tests fehlgeschlagen.")

            save_results(i, RESULTS_DIR / f"iteration_{i:02d}", files, test_result, response_text)

        except Exception as e:
            print(f"Fehler: {e}")

    print(f"\nFertig. Erfolgsrate: {successful_iterations/ITERATIONS*100:.1f}%")
    restore_project(backup_dir, PROJECT_DIR)
    write_summary(f"\nFertig. Erfolgsrate: {successful_iterations/ITERATIONS*100:.1f}%")

if __name__ == "__main__":
    main()