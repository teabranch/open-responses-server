#!/usr/bin/env python3
"""
Release script for openai-to-codex-wrapper.
This script:
1. Updates the version number in version.py
2. Creates a git commit and tag with the new version
3. Optionally builds and uploads the package to PyPI
"""

import argparse
import os
import re
import subprocess
import sys


def update_version(new_version):
    """Update the version in version.py"""
    version_file = "src/openai_to_codex_wrapper/version.py"
    
    with open(version_file, "r") as f:
        content = f.read()
    
    # Replace the version using regex
    updated_content = re.sub(r'__version__\s*=\s*"([^"]*)"', f'__version__ = "{new_version}"', content)
    
    with open(version_file, "w") as f:
        f.write(updated_content)
    
    print(f"Updated version to {new_version} in {version_file}")


def git_commit_and_tag(version):
    """Create a git commit and tag for the new version"""
    try:
        # Check if there are any changes to commit
        status = subprocess.run(["git", "status", "--porcelain"], 
                                capture_output=True, text=True, check=True)
        
        if not status.stdout.strip():
            print("No changes to commit. Did you already commit the version change?")
            return False
        
        # Commit version change
        subprocess.run(["git", "add", "src/openai_to_codex_wrapper/version.py"], check=True)
        subprocess.run(["git", "commit", "-m", f"Bump version to {version}"], check=True)
        
        # Create tag
        subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"Version {version}"], check=True)
        print(f"Created git tag v{version}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in git operations: {e}")
        return False


def build_and_publish():
    """Build and publish the package to PyPI"""
    try:
        # Clean up any existing build artifacts
        subprocess.run(["rm", "-rf", "dist", "build", "*.egg-info"], check=True)
        
        # Build the package
        subprocess.run(["python", "-m", "build"], check=True)
        
        # Upload to PyPI
        print("Uploading to PyPI...")
        subprocess.run(["python", "-m", "twine", "upload", "dist/*"], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building or publishing package: {e}")
        return False


def push_to_remote(version):
    """Push the new commit and tag to the remote repository"""
    try:
        push_changes = input("Push changes to remote? (y/n): ")
        if push_changes.lower() == 'y':
            subprocess.run(["git", "push"], check=True)
            subprocess.run(["git", "push", "origin", f"v{version}"], check=True)
            print("Pushed changes and tag to remote repository")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to remote: {e}")
        return False


def validate_version(version):
    """Validate that the version string is a valid semantic version"""
    pattern = r"^\d+\.\d+\.\d+$"
    if not re.match(pattern, version):
        print("Error: Version should be in the format X.Y.Z (e.g., 1.2.3)")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Release script for openai-to-codex-wrapper")
    parser.add_argument("version", help="New version number (e.g., 0.1.2)")
    parser.add_argument("--no-publish", action="store_true", help="Don't publish to PyPI")
    
    args = parser.parse_args()
    
    # Validate version format
    if not validate_version(args.version):
        return 1
    
    # Update version in file
    update_version(args.version)
    
    # Commit and tag
    if not git_commit_and_tag(args.version):
        print("Failed to create git commit and tag")
        return 1
    
    # Optionally build and publish
    if not args.no_publish:
        if not build_and_publish():
            print("Failed to build and publish package")
            return 1
    
    # Push to remote
    push_to_remote(args.version)
    
    print(f"Successfully released version {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())