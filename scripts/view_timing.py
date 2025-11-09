#!/usr/bin/env python3
"""
View and analyze timing data saved during training runs.

Usage:
    # View latest timing data
    python scripts/view_timing.py
    
    # View specific date
    python scripts/view_timing.py tmp/timing/timing_20251107.jsonl
    
    # Show detailed breakdown
    python scripts/view_timing.py --detailed
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics


def load_timing_data(timing_file: Path) -> List[dict]:
    """Load timing data from JSONL file."""
    data = []
    with open(timing_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_timing(data: List[dict], detailed: bool = False) -> None:
    """Analyze and print timing statistics."""
    if not data:
        print("No timing data found!")
        return
    
    # Group by operation
    by_operation = defaultdict(list)
    by_instance = defaultdict(lambda: defaultdict(list))
    
    for entry in data:
        operation = entry.get("operation", "unknown")
        duration = entry.get("duration_seconds", 0)
        instance_id = entry.get("instance_id", "unknown")
        
        by_operation[operation].append(duration)
        by_instance[instance_id][operation].append(duration)
    
    print("\n" + "="*80)
    print(f"TIMING SUMMARY ({len(data)} operations)")
    print("="*80 + "\n")
    
    # Overall statistics
    print(f"{'Operation':<20} {'Count':>8} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Total':>12}")
    print("-" * 80)
    
    for operation in sorted(by_operation.keys()):
        durations = by_operation[operation]
        count = len(durations)
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        min_val = min(durations)
        max_val = max(durations)
        total = sum(durations)
        
        print(f"{operation:<20} {count:>8} {mean:>9.2f}s {median:>9.2f}s {min_val:>9.2f}s {max_val:>9.2f}s {total:>11.1f}s")
    
    # Outliers
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS (operations > 2x median)")
    print("="*80 + "\n")
    
    for operation in sorted(by_operation.keys()):
        durations = by_operation[operation]
        if len(durations) < 2:
            continue
        
        median = statistics.median(durations)
        outliers = [d for d in durations if d > 2 * median]
        
        if outliers:
            print(f"{operation}:")
            print(f"  Median: {median:.2f}s, Outlier threshold: >{2*median:.2f}s")
            print(f"  Outliers: {len(outliers)}/{len(durations)} ({100*len(outliers)/len(durations):.1f}%)")
            print(f"  Slowest: {', '.join(f'{x:.1f}s' for x in sorted(outliers, reverse=True)[:5])}")
            print()
    
    # Per-instance breakdown if detailed
    if detailed:
        print("\n" + "="*80)
        print("PER-INSTANCE BREAKDOWN")
        print("="*80 + "\n")
        
        for instance_id in sorted(by_instance.keys()):
            ops = by_instance[instance_id]
            print(f"\nInstance: {instance_id}")
            for op in sorted(ops.keys()):
                durations = ops[op]
                print(f"  {op}: {len(durations)} calls, {sum(durations):.1f}s total (avg {statistics.mean(durations):.2f}s)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="View SWEbench timing data")
    parser.add_argument("file", nargs="?", help="Timing file to analyze (default: latest in tmp/timing/)")
    parser.add_argument("--detailed", action="store_true", help="Show per-instance breakdown")
    args = parser.parse_args()
    
    if args.file:
        timing_file = Path(args.file)
    else:
        # Find latest timing file
        timing_dir = Path("tmp/timing")
        if not timing_dir.exists():
            print("No timing directory found. Run training first!")
            sys.exit(1)
        
        timing_files = sorted(timing_dir.glob("timing_*.jsonl"))
        if not timing_files:
            print("No timing files found in tmp/timing/")
            sys.exit(1)
        
        timing_file = timing_files[-1]
        print(f"Using latest timing file: {timing_file}")
    
    if not timing_file.exists():
        print(f"File not found: {timing_file}")
        sys.exit(1)
    
    data = load_timing_data(timing_file)
    analyze_timing(data, detailed=args.detailed)


if __name__ == "__main__":
    main()

