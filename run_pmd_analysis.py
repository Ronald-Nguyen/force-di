import os
import subprocess
import xml.etree.ElementTree as ET
import re

# Configuration
PATH = "./force-di/main"
RULESET_FILE = "pmd-ruleset.xml"
REPORT_FILE = "pmd-report.xml"
SUMMARY_FILE = "complexity-summary.txt"

def create_ruleset_file(filename=RULESET_FILE):
    """Creates the PMD ruleset XML file."""
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
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {filename}")

def run_pmd_command(src_dir=PATH, ruleset=RULESET_FILE, report_file=REPORT_FILE):
    """Executes the PMD check command."""
    command = [
        "pmd", "check",
        "--dir", src_dir,
        "--rulesets", ruleset,
        "--format", "xml",
        "--report-file", report_file
    ]
    
    print(f"Executing PMD analysis on {src_dir}...")
    
    try:
        # shell=True is needed on Windows if 'pmd' is a batch file/wrapper
        subprocess.run(command, check=True, shell=True)
        print(f"Analysis complete. Report generated at {report_file}")
    except subprocess.CalledProcessError as e:
        if e.returncode > 0:
            print(f"PMD found violations (Exit code {e.returncode}). Proceeding...")
        else:
            print(f"Error running PMD: {e}")

def count_lines_in_file(filepath):
    """Reads a file and returns total lines."""
    try:
        if not os.path.exists(filepath):
            return 0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def get_all_project_files(src_dir):
    """Finds all .cls files and counts their lines."""
    file_stats = {}
    for root, _, files in os.walk(src_dir):
        for name in files:
            if not name.endswith('.cls'):
                continue
            filepath = os.path.join(root, name)
            abs_path = os.path.abspath(filepath)
            
            file_stats[abs_path] = {
                "relative_path": filepath,
                "loc": count_lines_in_file(filepath),
                "complexity": 0,
                "is_test": "test" in name.lower() # Case-insensitive check
            }
    return file_stats

def parse_and_summarize(src_dir=PATH, report_file=REPORT_FILE, output_file=SUMMARY_FILE):
    """Combines directory scan with PMD report and splits by Test/Prod."""
    
    all_files_map = get_all_project_files(src_dir)
    
    if os.path.exists(report_file):
        tree = ET.parse(report_file)
        root = tree.getroot()
        ns = {'pmd': 'http://pmd.sourceforge.net/report/2.0.0'}
        complexity_pattern = re.compile(r"cognitive complexity of (\d+)")

        for file_elem in root.findall('pmd:file', ns):
            report_path = file_elem.get('name')
            abs_report_path = os.path.abspath(report_path)
            
            total_complexity = 0
            for violation in file_elem.findall('pmd:violation', ns):
                if violation.get('rule') == 'CognitiveComplexity':
                    match = complexity_pattern.search(violation.text or "")
                    if match:
                        total_complexity += int(match.group(1))
            
            if abs_report_path in all_files_map:
                all_files_map[abs_report_path]['complexity'] = total_complexity
            else:
                # Fallback if a file was found by PMD but missed by scanner
                all_files_map[abs_report_path] = {
                    "relative_path": report_path,
                    "loc": count_lines_in_file(report_path),
                    "complexity": total_complexity,
                    "is_test": "test" in os.path.basename(report_path).lower()
                }

    # Statistics accumulators
    stats = {
        "prod": {"complexity": 0, "loc": 0, "count": 0},
        "test": {"complexity": 0, "loc": 0, "count": 0}
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Project Structure & Complexity Report\n")
        f.write("=====================================\n")
        f.write(f"{'Type':<6} | {'File Path':<72} | {'Compl.':<6} | {'LOC'}\n")
        f.write("-" * 100 + "\n")
        
        sorted_files = sorted(all_files_map.values(), key=lambda x: x['relative_path'])
        
        for data in sorted_files:
            category = "test" if data['is_test'] else "prod"
            type_label = "[TEST]" if data['is_test'] else "[PROD]"
            
            f.write(f"{type_label:<6} | {data['relative_path']:<72} | {data['complexity']:<6} | {data['loc']}\n")
            
            stats[category]["complexity"] += data['complexity']
            stats[category]["loc"] += data['loc']
            stats[category]["count"] += 1
            
        f.write("-" * 100 + "\n")
        f.write(f"SUMMARY PROD  ({stats['prod']['count']} files): Complexity: {stats['prod']['complexity']:<6} | LOC: {stats['prod']['loc']}\n")
        f.write(f"SUMMARY TEST  ({stats['test']['count']} files): Complexity: {stats['test']['complexity']:<6} | LOC: {stats['test']['loc']}\n")
        f.write(f"GRAND TOTAL   ({len(all_files_map)} files): Complexity: {stats['prod']['complexity'] + stats['test']['complexity']:<6} | LOC: {stats['prod']['loc'] + stats['test']['loc']}\n")

    print(f"Summary generated at {output_file}")

if __name__ == "__main__":
    create_ruleset_file()
    if not os.path.exists(PATH):
        print(f"Warning: Directory '{PATH}' does not exist.")
    else:
        run_pmd_command()
        parse_and_summarize()