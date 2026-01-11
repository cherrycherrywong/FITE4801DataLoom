Param(
    [string]$RemoteUrl = "https://github.com/cherrycherrywong/FITE4801DataLoom.git",
    [string]$Branch = "main",
    [string]$CommitMessage = "Initial commit: add project files"
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is not installed or not in PATH. Install git before running this script."
    exit 1
}

Write-Host "Initializing git repository and pushing to $RemoteUrl on branch $Branch"

if (-not (Test-Path .git)) {
    git init
}

git add --all
git commit -m "$CommitMessage" -q

try {
    git remote add origin $RemoteUrl -q
} catch {
    Write-Host "Remote 'origin' already exists — updating URL"
    git remote set-url origin $RemoteUrl
}

git branch -M $Branch

Write-Host "Pushing to remote... you may be prompted for credentials or use your Git credential manager"
git push -u origin $Branch

Write-Host "Done. If push failed, ensure the remote URL exists and your credentials are correct."
