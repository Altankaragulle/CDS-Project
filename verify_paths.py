"""
Quick script to verify all experiment paths are correct.
Run this to check if paths will work in Nuvolos.
"""

from pathlib import Path
import re

def check_experiment_paths():
    """Check all experiment files for correct path resolution."""
    
    experiments_dir = Path(__file__).parent / 'experiments'
    issues = []
    correct = []
    
    print("="*80)
    print("VERIFYING EXPERIMENT PATHS FOR NUVOLOS COMPATIBILITY")
    print("="*80)
    
    # Find all experiment Python files
    exp_files = sorted(experiments_dir.glob('exp*.py'))
    
    for exp_file in exp_files:
        with open(exp_file, 'r') as f:
            content = f.read()
        
        # Check for incorrect pattern
        if 'Path(__file__).parent.parent.parent' in content:
            issues.append((exp_file.name, "Uses .parent.parent.parent (too many levels!)"))
        # Check for relative paths without PROJECT_ROOT
        elif re.search(r"Path\(['\"]output['\"]", content) or re.search(r"Path\(['\"]report", content):
            issues.append((exp_file.name, "Uses relative paths without PROJECT_ROOT"))
        # Check for correct pattern
        elif 'Path(__file__).parent.parent' in content:
            correct.append(exp_file.name)
        else:
            issues.append((exp_file.name, "No PROJECT_ROOT definition found"))
    
    # Print results
    print(f"\n✅ CORRECT ({len(correct)} files):")
    print("-" * 80)
    for file in correct:
        print(f"  ✓ {file}")
    
    if issues:
        print(f"\n❌ ISSUES FOUND ({len(issues)} files):")
        print("-" * 80)
        for file, issue in issues:
            print(f"  ✗ {file}: {issue}")
        print("\n⚠️  These files will NOT work in Nuvolos!")
        return False
    else:
        print(f"\n🎉 ALL EXPERIMENTS HAVE CORRECT PATHS!")
        print("✅ Your project is ready for Nuvolos!")
        return True

def check_src_paths():
    """Check all src files for correct path resolution."""
    
    src_dir = Path(__file__).parent / 'src'
    issues = []
    correct = []
    
    print("\n" + "="*80)
    print("VERIFYING SRC PATHS")
    print("="*80)
    
    # Find all src Python files
    src_files = sorted(src_dir.glob('step*.py'))
    
    for src_file in src_files:
        with open(src_file, 'r') as f:
            content = f.read()
        
        # Check for correct pattern
        if 'Path(__file__).parent.parent' in content:
            correct.append(src_file.name)
        else:
            issues.append((src_file.name, "No PROJECT_ROOT definition found"))
    
    # Print results
    print(f"\n✅ CORRECT ({len(correct)} files):")
    print("-" * 80)
    for file in correct:
        print(f"  ✓ {file}")
    
    if issues:
        print(f"\n⚠️  POTENTIAL ISSUES ({len(issues)} files):")
        print("-" * 80)
        for file, issue in issues:
            print(f"  ? {file}: {issue}")
    
    return len(issues) == 0

def main():
    """Run all checks."""
    exp_ok = check_experiment_paths()
    src_ok = check_src_paths()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if exp_ok and src_ok:
        print("\n✅ ALL PATHS ARE CORRECT!")
        print("🎉 Your project is ready for Nuvolos!")
        print("\nNext steps:")
        print("  1. Upload project to Nuvolos")
        print("  2. Run: python3 main.py")
        print("  3. Verify experiments complete successfully")
    else:
        print("\n⚠️  SOME ISSUES FOUND")
        print("Please fix the issues listed above before uploading to Nuvolos.")
    
    print()

if __name__ == '__main__':
    main()
