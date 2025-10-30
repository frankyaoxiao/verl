#!/usr/bin/env python3
"""
Test to verify if shell state persists across multiple E2B command executions.
This will tell us if cd, environment variables, etc. persist.
"""

import json
import os
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox as E2BSandbox

# Load environment variables from .env file
load_dotenv()


def test_shell_persistence():
    """Test if shell state persists across multiple command calls."""
    
    print("=" * 80)
    print("E2B Shell Persistence Test")
    print("=" * 80)
    
    # Create E2B sandbox (matching how swebench_sandbox.py does it)
    sandbox = E2BSandbox.create(template="swebench-conda", timeout=600)
    
    try:
        print("\n📦 Sandbox created\n")
        
        # Test 1: Check initial directory (using exact format from swebench_sandbox.py)
        print("Test 1: Initial working directory")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'pwd'", timeout=60)
        initial_dir = result.stdout.strip()
        print(f"Initial directory: {initial_dir}")
        
        # Test 2: Change directory and check
        print("\nTest 2: Change directory in first command")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'cd /tmp && pwd'", timeout=60)
        dir_during_cd = result.stdout.strip()
        print(f"Directory during cd command: {dir_during_cd}")
        
        # Test 3: Check directory in next command (THIS IS THE KEY TEST)
        print("\nTest 3: Check directory in NEXT command")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'pwd'", timeout=60)
        dir_after_cd = result.stdout.strip()
        print(f"Directory after cd command: {dir_after_cd}")
        
        if dir_after_cd == "/tmp":
            print("✅ PERSISTENT: Working directory persisted!")
        else:
            print("❌ NOT PERSISTENT: Working directory reset to:", dir_after_cd)
        
        # Test 4: Environment variables
        print("\nTest 4: Environment variable persistence")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'export MY_VAR=hello && echo $MY_VAR'", timeout=60)
        var_during_export = result.stdout.strip()
        print(f"Variable during export: {var_during_export}")
        
        result = sandbox.commands.run("bash -lc 'echo $MY_VAR'", timeout=60)
        var_after_export = result.stdout.strip()
        print(f"Variable in next command: '{var_after_export}'")
        
        if var_after_export == "hello":
            print("✅ PERSISTENT: Environment variable persisted!")
        else:
            print("❌ NOT PERSISTENT: Environment variable lost")
        
        # Test 5: Filesystem changes (should persist)
        print("\nTest 5: Filesystem persistence")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'echo test > /tmp/testfile.txt'", timeout=60)
        print("Created /tmp/testfile.txt")
        
        result = sandbox.commands.run("bash -lc 'cat /tmp/testfile.txt'", timeout=60)
        file_content = result.stdout.strip()
        print(f"File content in next command: {file_content}")
        
        if file_content == "test":
            print("✅ PERSISTENT: Filesystem changes persisted!")
        else:
            print("❌ NOT PERSISTENT: File was lost")
        
        # Test 6: Using cd in the same command with &&
        print("\nTest 6: cd with && in single command")
        print("-" * 40)
        result = sandbox.commands.run("bash -lc 'cd /usr && pwd && ls bin | head -5'", timeout=60)
        output = result.stdout.strip()
        print(f"Output:\n{output}")
        if "/usr" in output:
            print("✅ cd with && works within a single command")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Initial directory: {initial_dir}")
        print(f"After cd: {dir_after_cd}")
        print(f"Verdict: {'Shell state is NOT persistent across commands' if dir_after_cd == initial_dir else 'Shell state IS persistent'}")
        
    finally:
        sandbox.kill()
        print("\n🧹 Sandbox closed")


if __name__ == "__main__":
    test_shell_persistence()

