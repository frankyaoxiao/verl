#!/usr/bin/env python3
"""
Parse TIMING logs from training runs and generate summary statistics.

Usage:
    python scripts/analyze_timing_logs.py <log_file_or_directory>
    
    # Parse a single log file
    python scripts/analyze_timing_logs.py training_output.log
    
    # Parse all logs in a directory
    python scripts/analyze_timing_logs.py tmp/verl_swebench_logs/
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics


def parse_timing_logs(log_path: Path) -> Dict[str, List[float]]:
    """Parse TIMING logs and extract operation durations."""
    timings = defaultdict(list)
    
    # Pattern: [TIMING] <id> - Completed <operation> in <duration>s
    pattern = r'\[TIMING\] \w+ - Completed (\w+) in ([\d.]+)s'
    
    if log_path.is_file():
        files = [log_path]
    else:
        files = list(log_path.glob("**/*.log")) + list(log_path.glob("**/*"))
        files = [f for f in files if f.is_file()]
    
    for file in files:
        try:
            with open(file, 'r', errors='ignore') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        operation = match.group(1)
                        duration = float(match.group(2))
                        timings[operation].append(duration)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}", file=sys.stderr)
    
    return timings


def print_statistics(timings: Dict[str, List[float]]) -> None:
    """Print summary statistics for each operation."""
    if not timings:
        print("No timing data found!")
        return
    
    print("\n" + "="*80)
    print("TIMING ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    # Sort by total time spent (count * mean)
    operations = sorted(
        timings.items(), 
        key=lambda x: len(x[1]) * statistics.mean(x[1]),
        reverse=True
    )
    
    print(f"{'Operation':<25} {'Count':>8} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Total':>10}")
    print("-" * 80)
    
    for operation, durations in operations:
        count = len(durations)
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        min_val = min(durations)
        max_val = max(durations)
        total = sum(durations)
        
        print(f"{operation:<25} {count:>8} {mean:>9.2f}s {median:>9.2f}s {min_val:>9.2f}s {max_val:>9.2f}s {total:>9.1f}s")
    
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS (operations > 2x median)")
    print("="*80 + "\n")
    
    for operation, durations in operations:
        if len(durations) < 2:
            continue
        
        median = statistics.median(durations)
        outliers = [d for d in durations if d > 2 * median]
        
        if outliers:
            print(f"{operation}:")
            print(f"  Median: {median:.2f}s")
            print(f"  Outliers (>{2*median:.2f}s): {len(outliers)}/{len(durations)}")
            print(f"  Outlier values: {', '.join(f'{x:.1f}s' for x in sorted(outliers, reverse=True)[:5])}")
            if len(outliers) > 5:
                print(f"  ... and {len(outliers)-5} more")
            print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    
    if not log_path.exists():
        print(f"Error: {log_path} does not exist")
        sys.exit(1)
    
    print(f"Analyzing timing logs in: {log_path}")
    timings = parse_timing_logs(log_path)
    print_statistics(timings)


if __name__ == "__main__":
    main()


