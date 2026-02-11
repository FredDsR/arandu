Create a pull request from the current branch to the main branch using the GitHub MCP tools.

Steps:
1. Run `git branch --show-current` to get the current branch name.
2. Run `git remote get-url origin` to extract the repository owner and name.
3. Run `git status --short` to check for uncommitted changes. If there are uncommitted changes, warn the user and stop — do NOT commit or push on their behalf.
4. Run `git log --oneline main..HEAD` to collect all commits on the current branch since it diverged from main.
5. Run `git diff main..HEAD --stat` to see the summary of changed files.
6. Check if the branch has been pushed to the remote by running `git ls-remote --heads origin <branch>`. If not pushed, inform the user they need to push first and stop.
7. Analyze the commits and changed files to draft a PR title and body:
   - The title should follow conventional commit style: `<type>(<scope>): <description>` (under 70 characters).
   - The body should include a `## Summary` section with bullet points explaining the changes.
   - The body should include a `## Test plan` section with a checklist of testing steps.
8. Show the drafted title and body to the user for confirmation before creating.
9. Use the GitHub MCP `create_pull_request` tool to create the PR with the confirmed title and body.
10. Return the PR URL to the user.

Rules:
- Use types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- Scope is optional but encouraged (e.g., cep, cli, qa, transcribe)
- Keep the PR title under 70 characters
- Use imperative mood ("add feature" not "added feature")
- Do NOT commit, push, or merge — only create the pull request
- The base branch is always `main` unless the user specifies otherwise via arguments
- If $ARGUMENTS is provided and looks like a branch name, use it as the base branch instead of `main`
- Do not self-reference Claude as a co-author
- **CRITICAL**: The PR body MUST use actual newlines, NOT literal `\n` escape sequences. Do NOT pass the body as a single-line string with `\n` — use real multi-line text so that Markdown renders correctly on GitHub
