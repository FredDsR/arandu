Generate a conventional commit message for the current staged changes, then copy it to the system clipboard.

Steps:
1. Run `git diff --stat` and `git diff` to see all changes (staged + unstaged).
2. Run `git log --oneline -5` to see recent commit style.
3. Analyze the changes and draft a commit message following Conventional Commits format: `<type>(<scope>): <description>`.
4. The message body should summarize the "why" with concise bullet points when multiple changes are involved. Each bullet point must be a single unwrapped line — do not insert line breaks within the same paragraph or bullet.
5. Copy the full message to the clipboard using `xclip -selection clipboard`.
6. Show the commit message to the user and confirm it was copied.

Rules:
- Use types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
- Scope is optional but encouraged (e.g., versioning, cli, batch, qa)
- Keep the subject line under 72 characters
- Use imperative mood ("add feature" not "added feature")
- Do NOT create a commit, only generate the message and copy it
- Do not self-reference Claude as a co-author
