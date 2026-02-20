Fetch all review comments from a pull request, implement the requested fixes, and reply to each comment confirming the resolution.

The PR number is: $ARGUMENTS

Steps:
1. If no PR number was provided, ask the user for the PR number before proceeding.
2. Run `git remote get-url origin` to extract the repository owner and name.
3. Use the GitHub MCP `pull_request_read` tool (method: `get`) to fetch the PR title, head branch, and base branch.
4. Run `git branch --show-current` and verify you are on the PR's head branch. If not, warn the user and stop.
5. Use the GitHub MCP `pull_request_read` tool (method: `get_review_comments`) to fetch all review threads and their comments.
6. Group and categorize the comments into actionable items. For each comment, note:
   - The comment ID (needed for replying later)
   - The file path and line number
   - The requested change or suggestion
   - Whether it is already resolved/outdated
7. Present a numbered summary table of all comments to the user, showing: file, issue summary, and proposed fix. Ask the user to confirm which comments to address (default: all unresolved).
8. For each confirmed comment, implement the fix:
   - Read the relevant file before making changes.
   - Apply the minimal change that addresses the feedback.
   - Follow all project coding standards from AGENT.md (type annotations, Google docstrings, Ruff compliance, etc.).
9. After all fixes are applied:
   - Run `ruff check --fix` and `ruff format` on all modified source and test files.
   - Run `pytest` on any affected test directories.
   - If tests or linting fail, fix the issues before proceeding.
10. Stage all changed files and commit using conventional commit format: `fix(<scope>): address PR #<number> review comments`.
11. Push the commit to the remote.
12. For each addressed comment, use the GitHub MCP `add_reply_to_pull_request_comment` tool to reply with a short confirmation message that references the fixing commit SHA and describes what was changed. Use the comment ID captured in step 6.
13. Print a final summary showing how many comments were addressed, the commit SHA, and any comments that were skipped (with reasons).

Rules:
- ALWAYS use GitHub MCP tools for all GitHub interactions (reading PRs, fetching comments, replying). Never use the `gh` CLI as a fallback.
- Do NOT self-reference as Claude, AI, or assistant in commit messages, PR replies, or any generated text. Write as if the developer authored it directly.
- Do NOT add co-author trailers or AI attribution to commits.
- Do NOT resolve/dismiss review threads — only reply to them. The reviewer decides when to resolve.
- Do NOT modify files unrelated to the review comments.
- Do NOT create new tests unless a comment specifically requests test coverage.
- Do NOT push to main or change the base branch.
- Keep replies concise: one or two sentences referencing the commit and what was fixed.
- If a comment is ambiguous or requires a design decision, ask the user before implementing.
- If a comment suggests a code snippet (```suggestion``` block), apply it exactly unless it conflicts with project standards.
- Group related comments that affect the same file into a single logical change when possible.
- Always read a file before editing it.
