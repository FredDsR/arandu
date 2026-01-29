# CI/CD Setup Guide

This document explains how to set up the GitHub Actions workflow for automated linting, testing, and badge generation.

## Overview

The CI workflow (`.github/workflows/ci.yml`) automatically:
- Runs Ruff linting on all Python code
- Runs pytest with coverage reporting
- Enforces minimum 75% test coverage
- Updates dynamic badges for test status and coverage percentage

## Workflow Triggers

The workflow runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger via `workflow_dispatch`

## Jobs

### 1. Lint Job
- Runs Ruff linter (`ruff check`)
- Runs Ruff formatter check (`ruff format --check`)
- Fails if any linting issues are found

### 2. Test Job
- Runs pytest with coverage
- Requires minimum 75% coverage (fails if below)
- Extracts test count and coverage percentage
- Updates dynamic badges (on `main` branch only)

## Badge Setup

The workflow uses dynamic badges that update automatically with each CI run. This requires a one-time setup:

### Step 1: Create a GitHub Gist

1. Go to https://gist.github.com/
2. Create a new **public** gist with any filename (e.g., `badges.md`)
3. Add minimal content (e.g., `# Badges`)
4. Create the gist and note the gist ID from the URL
   - URL format: `https://gist.github.com/USERNAME/GIST_ID`
   - Example: If URL is `https://gist.github.com/FredDsR/abc123def456`, the gist ID is `abc123def456`

### Step 2: Create a Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name (e.g., "CI Badge Updates")
4. Select scope: **gist** (only this scope is needed)
5. Generate token and copy it immediately (you won't see it again)

### Step 3: Add Repository Secrets

1. Go to your repository → Settings → Secrets and variables → Actions
2. Add two new repository secrets:
   - **Name**: `GIST_SECRET`
     - **Value**: The personal access token from Step 2
   - **Name**: `GIST_ID`
     - **Value**: The gist ID from Step 1

### Step 4: Update README Badges

Replace `GIST_ID` in the README.md badge URLs with your actual gist ID:

```markdown
![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/YOUR_GIST_ID/raw/tests-badge.json)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/YOUR_GIST_ID/raw/coverage-badge.json)
```

Replace `YOUR_GIST_ID` with the actual gist ID from Step 1.

## How It Works

1. On every push/PR, the workflow runs linting and tests
2. If tests pass, it extracts:
   - Total number of tests
   - Coverage percentage
3. On `main` branch, it updates two JSON files in the gist:
   - `tests-badge.json`: Test count badge data
   - `coverage-badge.json`: Coverage percentage badge data
4. Badges in README automatically display the latest values

## Coverage Badge Colors

The coverage badge automatically changes color based on percentage:
- 🔴 Red: 0-59%
- 🟡 Yellow: 60-74%
- 🟢 Green: 75-100%

## Local Testing

Before pushing, you can run the same checks locally:

```bash
# Lint check
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Run tests with coverage
uv run pytest --cov=gtranscriber --cov-report=term-missing --cov-fail-under=75
```

## Troubleshooting

### Badges not updating

1. Verify the workflow ran successfully (check Actions tab)
2. Check that secrets `GIST_SECRET` and `GIST_ID` are set correctly
3. Ensure the gist is **public** (private gists won't work with shields.io)
4. Wait a few minutes for shields.io cache to refresh

### Coverage failing

If coverage drops below 75%, the workflow will fail. To fix:
1. Add more tests to increase coverage
2. Or adjust the threshold in `.github/workflows/ci.yml` (line with `--cov-fail-under=75`)

### Linting failures

If linting fails:
1. Run `uv run ruff check --fix src/ tests/` to auto-fix issues
2. Run `uv run ruff format src/ tests/` to format code
3. Commit the changes

## Workflow File Location

`.github/workflows/ci.yml`

## Required Dependencies

The workflow uses:
- `astral-sh/setup-uv@v5` - Install uv package manager
- `actions/setup-python@v5` - Install Python 3.13
- `schneegans/dynamic-badges-action@v1.7.0` - Update gist badges

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Dynamic Badges Action](https://github.com/Schneegans/dynamic-badges-action)
- [Shields.io Endpoint Badges](https://shields.io/badges/endpoint-badge)
