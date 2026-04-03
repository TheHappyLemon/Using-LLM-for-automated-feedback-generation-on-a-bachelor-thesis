import os
import re
import itertools
import subprocess
from pathlib import Path
from src.data.constants import BASE_PATH

ROOT_DIR =  Path(os.path.join(BASE_PATH, "src", "results", "llm", "zero_temperature_determinism", "responses", "gpt-oss-20b-thinking", "t0"))
REPORT_FILE = Path("./diff_report.txt")
TARGET_TYPES = ["BeforeGoal", "AfterTasks", "Goal", "Tasks"]
# 10_BeforeGoal_gpt-oss-20b-thinking_t0_1.json
FILENAME_RE = re.compile(
    r"^\d+_(BeforeGoal|AfterTasks|Goal|Tasks)_.+_t\d+_(\d+)\.json$"
)


def find_files(root_dir: Path):
    """
    Build a structure like:
    {
        "BeforeGoal": { "1": Path(...), "2": Path(...), ... },
        "AfterTasks": { ... },
        ...
    }
    """
    files_by_type = {t: {} for t in TARGET_TYPES}
    for entry in root_dir.iterdir():
        if not entry.is_dir():
            continue

        iteration = entry.name  # folder name: 1, 2, 3, ...

        for file in entry.iterdir():
            if not file.is_file():
                continue

            match = FILENAME_RE.match(file.name)
            if not match:
                continue

            file_type = match.groups()
            files_by_type[file_type[0]][iteration] = file
            
    return files_by_type


def run_diff_quiet(file1: Path, file2: Path):
    """
    Uses: diff -q file1 file2
    Returns:
        "same"      -> files are identical
        "different" -> files differ
        "error"     -> diff command failed
    """
    try:
        result = subprocess.run(
            ["diff", "-q", str(file1), str(file2)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return "same", result.stdout.strip(), result.stderr.strip()
        elif result.returncode == 1:
            return "different", result.stdout.strip(), result.stderr.strip()
        else:
            return "error", result.stdout.strip(), result.stderr.strip()

    except FileNotFoundError:
        return "error", "", "The 'diff' command was not found in PATH."


def numeric_sort_key(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    files_by_type = find_files(ROOT_DIR)
    total_comparisons = 0
    same_count = 0
    different_count = 0
    error_count = 0

    lines = []
    lines.append(f"Root directory: {ROOT_DIR}")
    lines.append("Comparison rule: only same-type files across iteration folders")
    lines.append("Compared types: BeforeGoal, AfterTasks, Goal, Tasks")
    lines.append("Method: diff -q")
    lines.append("")

    for file_type in TARGET_TYPES:
        lines.append("=" * 80)
        lines.append(f"TYPE: {file_type}")
        lines.append("=" * 80)

        iteration_map = files_by_type[file_type]
        iterations = sorted(iteration_map.keys(), key=numeric_sort_key)

        if len(iterations) < 2:
            lines.append("Not enough files to compare.")
            lines.append("")
            continue

        lines.append("Files found:")
        for it in iterations:
            lines.append(f"  Iteration {it}: {iteration_map[it]}")
        lines.append("")

        lines.append("Comparisons:")
        for it1, it2 in itertools.combinations(iterations, 2):
            file1 = iteration_map[it1]
            file2 = iteration_map[it2]

            status, stdout_text, stderr_text = run_diff_quiet(file1, file2)
            total_comparisons += 1

            if status == "same":
                same_count += 1
                lines.append(
                    f"[SAME] Iteration {it1} vs {it2} | "
                    f"{file1.name} == {file2.name}"
                )
            elif status == "different":
                different_count += 1
                lines.append(
                    f"[DIFFERENT] Iteration {it1} vs {it2} | "
                    f"{file1.name} != {file2.name}"
                )
                #if stdout_text:
                #    lines.append(f"  diff output: {stdout_text}")
            else:
                error_count += 1
                lines.append(
                    f"[ERROR] Iteration {it1} vs {it2} | "
                    f"{file1.name} vs {file2.name}"
                )
                if stdout_text:
                    lines.append(f"  stdout: {stdout_text}")
                if stderr_text:
                    lines.append(f"  stderr: {stderr_text}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total comparisons: {total_comparisons}")
    lines.append(f"Same: {same_count}")
    lines.append(f"Different: {different_count}")
    lines.append(f"Errors: {error_count}")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()