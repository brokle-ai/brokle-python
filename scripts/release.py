#!/usr/bin/env python3
"""
Brokle SDK Release Script

Automates the release process for the Brokle Python SDK.
Handles version bumping, testing, building, and publishing.
"""

import subprocess
import sys
import argparse
import logging
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="🚀 Brokle Release - %(message)s")


def run_command(command, check=True):
    logging.info(f"Running command: {command}")
    result = subprocess.run(
        command, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def check_git_status():
    """Check git status and ensure we're ready for release."""
    # Check if there are uncommitted changes
    logging.info("Checking for uncommitted changes...")
    status_output = run_command("git status --porcelain", check=False)
    if status_output:
        logging.error(
            "Your working directory has uncommitted changes. Please commit or stash them before proceeding."
        )
        sys.exit(1)

    # Check if the current branch is 'main'
    logging.info("Checking the current branch...")
    current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    if current_branch != "main":
        logging.error(
            "You are not on the 'main' branch. Please switch to 'main' before proceeding."
        )
        sys.exit(1)

    # Pull the latest changes from remote 'main'
    logging.info("Pulling the latest changes from remote 'main'...")
    run_command("git pull origin main")


def get_latest_tag():
    """Get the latest git tag."""
    try:
        latest_tag = run_command("git describe --tags --abbrev=0")
        if latest_tag.startswith("v"):
            latest_tag = latest_tag[1:]
    except subprocess.CalledProcessError:
        latest_tag = "0.0.0"  # default if no tags exist
    return latest_tag


def increment_version(current_version, increment_type):
    """Increment version based on type (major, minor, patch)."""
    # Check if current version is a pre-release
    if "-" in current_version:
        # Pre-release detected - promote to stable (strip suffix, don't increment)
        base_version = current_version.split("-")[0]
        logging.info(f"Pre-release detected: {current_version}")
        logging.info(f"Promoting to stable: {base_version}")
        return base_version

    # Stable version - increment normally
    major, minor, patch = map(int, current_version.split("."))
    if increment_type == "patch":
        patch += 1
    elif increment_type == "minor":
        minor += 1
        patch = 0
    elif increment_type == "major":
        major += 1
        minor = 0
        patch = 0
    return f"{major}.{minor}.{patch}"


def update_version_file(version):
    """Update the version in brokle/version.py."""
    version_file_path = "brokle/version.py"
    logging.info(f"Updating version in {version_file_path} to {version}...")

    with open(version_file_path, "r") as file:
        content = file.read()

    # Update __version__
    new_content = re.sub(
        r'__version__ = "\d+\.\d+\.\d+"', f'__version__ = "{version}"', content
    )

    # Update __version_info__
    major, minor, patch = version.split(".")
    new_content = re.sub(
        r'__version_info__ = \(\d+, \d+, \d+\)',
        f'__version_info__ = ({major}, {minor}, {patch})',
        new_content
    )

    with open(version_file_path, "w") as file:
        file.write(new_content)

    logging.info(f"Updated version in {version_file_path}.")


def run_tests():
    """Run the full test suite."""
    logging.info("Running test suite...")
    run_command("make test")
    logging.info("All tests passed.")



def main():
    parser = argparse.ArgumentParser(
        description="Automate the release process for the Brokle Python SDK."
    )
    parser.add_argument(
        "increment_type",
        choices=["patch", "minor", "major"],
        help="Specify the type of version increment.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests before release",
    )
    args = parser.parse_args()

    increment_type = args.increment_type

    try:
        logging.info("Starting Brokle SDK release process...")

        # Preliminary checks
        logging.info("Performing preliminary checks...")
        check_git_status()
        logging.info("Git status is clean, on 'main' branch, and up to date.")

        # Get the latest tag
        current_version = get_latest_tag()
        logging.info(f"Current version: v{current_version}")

        # Determine the new version
        new_version = increment_version(current_version, increment_type)
        logging.info(f"Proposed new version: v{new_version}")

        # Ask for user confirmation
        confirm = input(
            f"Do you want to proceed with the release version v{new_version}? (y/n): "
        )
        if confirm.lower() != "y":
            logging.info("Release process aborted by user.")
            sys.exit(0)

        # Step 1: Run tests (unless skipped)
        if not args.skip_tests:
            logging.info("Step 1: Running tests...")
            run_tests()
        else:
            logging.info("Step 1: Skipping tests (--skip-tests flag used)")

        # Step 2: Update the version
        logging.info("Step 2: Updating the version...")
        update_version_file(new_version)

        # Step 3: Build the package
        logging.info("Step 3: Building the package...")
        run_command("make clean && make build")

        # Ask for user confirmation
        confirm = input(
            f"Please check the changed files. Proceed with releasing v{new_version}? (y/n): "
        )
        if confirm.lower() != "y":
            logging.info("Release process aborted by user.")
            sys.exit(0)

        # Step 4: Commit the changes
        logging.info("Step 4: Committing the changes...")
        run_command("git add .")
        run_command(f'git commit -m "chore: bump version to {new_version}"')

        # Step 5: Push the commit
        logging.info("Step 5: Pushing the commit...")
        run_command("git push")

        # Step 6: Tag the version
        logging.info("Step 6: Tagging the version...")
        run_command(f"git tag v{new_version}")

        # Step 7: Push the tags
        logging.info("Step 7: Pushing the tags...")
        run_command("git push --tags")

        # Step 8: Success Message
        logging.info("Step 8: Release Completed")
        print("━" * 80)
        print("✅ Version v{} prepared and pushed!".format(new_version))
        print("━" * 80)
        print("")
        print("📋 NEXT STEPS:")
        print("")
        print("1. Go to: https://github.com/brokle-ai/brokle-python/releases/new")
        print(f"2. Select tag: v{new_version}")
        print("3. Click 'Generate release notes'")
        print("4. Review and edit release notes")
        print("5. Mark as pre-release if needed (for alpha/beta/rc)")
        print("6. Click 'Publish release'")
        print("")
        print("🤖 After you publish the release:")
        print("   - GitHub Actions will build the Python package")
        print("   - GitHub Actions will publish to PyPI (OIDC Trusted Publishing)")
        print("   - Artifacts will be uploaded to the release")
        print("")
        print("⏳ The release workflow will start automatically when you publish!")
        print("")
        logging.info("🚀 Brokle SDK release process completed successfully!")

    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running command: {e.cmd}")
        logging.error(e.stderr.decode("utf-8"))
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Release process interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()