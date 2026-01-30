# GitHub Workflow Setup - Next Steps

## What Was Created

### 1. CI Workflow (`.github/workflows/ci.yml`)
A GitHub Actions workflow that runs on every push and pull request to `main` or `develop` branches. It includes:

**Lint Job:**
- Runs Ruff linter to check code quality
- Runs Ruff formatter to ensure consistent code style
- Fails if any issues are found

**Test Job:**
- Runs pytest with coverage reporting
- **Enforces minimum 75% code coverage** - build fails if coverage is below this threshold
- Extracts test count and coverage percentage
- Updates dynamic badges automatically (only on `main` branch)

### 2. Updated README.md
- Added CI workflow status badge
- Changed static badges to dynamic badges that update automatically
- Badges will show real-time test count and coverage percentage

### 3. Setup Documentation (`.github/CI_SETUP.md`)
Comprehensive guide explaining:
- How the workflow works
- How to set up GitHub secrets for badge updates
- How to create and configure a GitHub Gist
- Troubleshooting tips
- Local testing commands

## Required Setup (One-Time Configuration)

To enable automatic badge updates, you need to:

### 1. Create a GitHub Gist
1. Go to https://gist.github.com/
2. Create a new **public** gist
3. Note the gist ID from the URL (e.g., `abc123def456`)

### 2. Create a Personal Access Token
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with **gist** scope only
3. Copy the token

### 3. Add Repository Secrets
1. Go to Repository Settings → Secrets and variables → Actions
2. Add two secrets:
   - `GIST_SECRET`: Your personal access token
   - `GIST_ID`: Your gist ID

### 4. Update README.md
Replace `GIST_ID` in README.md with your actual gist ID:
```markdown
![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/YOUR_GIST_ID/raw/tests-badge.json)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FredDsR/YOUR_GIST_ID/raw/coverage-badge.json)
```

## How It Works

1. **Every push/PR**: Workflow runs linting and tests
2. **Coverage check**: Build fails if coverage < 75%
3. **On main branch**: Badges are updated automatically in the gist
4. **Badges update**: README badges show latest values from the gist

## Local Testing

Before pushing changes, test locally:

```bash
# Check linting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Run tests with coverage
uv run pytest --cov=gtranscriber --cov-report=term-missing --cov-fail-under=75
```

## Coverage Badge Colors

The coverage badge color changes automatically:
- 🔴 **Red** (0-59%): Critical - add more tests
- 🟡 **Yellow** (60-74%): Warning - below threshold, build fails
- 🟢 **Green** (75-100%): Good - meets requirements

**Important**: The build fails if coverage is below 75% (enforced by `--cov-fail-under=75`). The badge uses floor division to display the percentage, ensuring it never shows a higher value than what actually passed.

## What Happens on Each Push

1. **Lint Job** runs first (fast feedback)
   - Checks code style with Ruff
   - Must pass or workflow fails

2. **Test Job** runs in parallel
   - Installs all dependencies
   - Runs pytest with coverage
   - **Fails if coverage < 75%**
   - Extracts metrics

3. **Badge Update** (main branch only)
   - Updates test count in gist
   - Updates coverage percentage in gist
   - Badges refresh within minutes

## Files Modified

- `.github/workflows/ci.yml` - GitHub Actions workflow
- `.github/CI_SETUP.md` - Setup documentation
- `README.md` - Badge URLs updated

## Testing the Workflow

After merging this PR:

1. The workflow will run automatically
2. Check the "Actions" tab to see the workflow execution
3. Verify both lint and test jobs pass
4. After setting up secrets, badges will update on the next merge to `main`

## Important Notes

- Badge updates only happen on `main` branch
- Secrets are **required** only for badge updates, not for the workflow itself
- The workflow will run and enforce coverage even without secrets
- Coverage threshold can be adjusted in the workflow file (line 61)

## See Also

- Full setup instructions: `.github/CI_SETUP.md`
- Workflow file: `.github/workflows/ci.yml`
- Project guidelines: `AGENTS.md`
