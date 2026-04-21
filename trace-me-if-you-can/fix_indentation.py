#!/usr/bin/env python3
import os
import sys
import ast
import textwrap

def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        except:
            return False

    original_content = content
    dedented_content = textwrap.dedent(content)
    
    if dedented_content == original_content:
        return False
    
    try:
        ast.parse(dedented_content)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dedented_content)
        return True
    except (IndentationError, SyntaxError):
        pass
    
    return False

def process_directory(root_dir):
    count = 0
    fixed = 0
    base_fixed = 0
    variant_fixed = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                count += 1

                if fix_file(filepath):
                    print(f"Fixed: {filepath}")
                    fixed += 1
                    if filename == 'base.py':
                        base_fixed += 1
                    else:
                        variant_fixed += 1

    print(f"Scanned {count} files")
    print(f"  Fixed base.py: {base_fixed}")
    print(f"  Fixed variant: {variant_fixed}")
    print(f"  Total fixed: {fixed}")

def main():
    root_dir = "/home/fahad/Documents/PROJECTS/Ajaira_Projects/trace-me-if-you-can/extracted_code_test"

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    process_directory(root_dir)

if __name__ == "__main__":
    main()