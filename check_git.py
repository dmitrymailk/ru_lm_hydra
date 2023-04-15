import subprocess

# https://github.com/roskakori/check_uncommitted_git_changes/blob/main/check_uncommitted_git_changes/main.py#L37
def has_uncommitted_git_changes() -> bool:
    git_output = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True)
    return len(git_output.stdout) > 0

print(has_uncommitted_git_changes())