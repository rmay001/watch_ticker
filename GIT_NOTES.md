# Git cheat sheet

Repo: https://github.com/rmay001/watch_ticker.git (branch `main`)

## Save and push changes
```bash
cd /Users/ronaldmay/Documents/Code
git status                      # see what changed
git add -A                      # stage everything (respects .gitignore)
git commit -m "describe the change"
git push
```

## Useful extras
```bash
git log --oneline               # history
git diff                        # unstaged changes
git pull                        # get changes from GitHub
git restore watch_timer.py      # discard uncommitted edits to a file
```

## One-time setup (already done)
```bash
git init
git branch -M main
git remote add origin https://github.com/rmay001/watch_ticker.git
git push -u origin main
```

## Login
HTTPS pushes need a personal access token as the password, or run `gh auth login`.
