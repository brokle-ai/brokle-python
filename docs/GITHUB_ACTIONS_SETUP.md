# GitHub Actions CI/CD Setup for Brokle Python SDK

I've created a complete CI/CD pipeline with 4 workflows. Here's how to set it up and use it:

## 🚀 Quick Setup (5 minutes)

### 1. Configure PyPI Secrets

You need to add these secrets to your GitHub repository:

#### A. Get PyPI API Tokens

1. **Test PyPI Token:**
   - Go to: https://test.pypi.org/account/register/
   - Create account → Go to Account Settings → API Tokens
   - Create token with scope: "Entire account"
   - Copy the token (starts with `pypi-`)

2. **Production PyPI Token:**
   - Go to: https://pypi.org/account/register/
   - Create account → Go to Account Settings → API Tokens
   - Create token with scope: "Entire account"
   - Copy the token (starts with `pypi-`)

#### B. Add Secrets to GitHub

1. Go to your repository on GitHub
2. Settings → Secrets and Variables → Actions
3. Add these Repository Secrets:

```
Name: TEST_PYPI_API_TOKEN
Value: pypi-AgEIcHlwaS5vcmcC... (your test PyPI token)

Name: PYPI_API_TOKEN
Value: pypi-AgEIcHlwaS5vcmcC... (your production PyPI token)
```

### 2. Push the Workflows

```bash
# The workflows are already created, just commit and push
cd sdk/python
git add .github/workflows/
git commit -m "feat: add GitHub Actions CI/CD pipeline"
git push origin main
```

## 🎯 How to Use the Workflows

### Option 1: Automatic Publishing (Recommended)

**For Alpha/Beta releases:**
```bash
# Create and push a tag
git tag v0.1.0a1
git push origin v0.1.0a1
```

**What happens:**
1. ✅ Tests run automatically
2. ✅ Package builds automatically
3. ✅ GitHub release created automatically
4. ✅ Published to PyPI automatically
5. ✅ Users can install: `pip install brokle==0.1.0a1`

### Option 2: Manual Publishing

**Publish to Test PyPI:**
1. Go to Actions tab in GitHub
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select "testpypi"
5. Click "Run workflow"

**Publish to Production PyPI:**
1. Same steps but select "pypi"

### Option 3: Development Testing

Every push to `main` automatically:
- ✅ Runs tests on Python 3.9, 3.10, 3.11, 3.12
- ✅ Checks code formatting
- ✅ Runs security scans
- ✅ Builds package
- ✅ Tests installation

## 📋 The 4 Workflows Explained

### 1. `ci.yml` - Continuous Integration
**Triggers:** Every push/PR
**What it does:**
- Tests on multiple Python versions
- Code quality checks (linting, formatting, type checking)
- Security scanning
- Build verification
- Coverage reporting

### 2. `publish.yml` - Manual Publishing
**Triggers:** Manual or on release
**What it does:**
- Builds package
- Publishes to Test PyPI or PyPI
- Manual control over publishing

### 3. `integration-test.yml` - Integration Testing
**Triggers:** Push to main, daily schedule
**What it does:**
- Integration tests with backends
- Package installation tests
- End-to-end functionality tests

### 4. `release.yml` - Automated Releases
**Triggers:** Git tags (v*.*.*)
**What it does:**
- Creates GitHub releases with auto-generated changelog
- Publishes to PyPI automatically
- Handles alpha/beta/rc versions
- Uploads wheel and source distributions

## 🚀 Quick Start - Publish Your First Alpha

```bash
# 1. Add secrets to GitHub (see above)

# 2. Commit workflows
git add .github/workflows/
git commit -m "feat: add CI/CD pipeline"
git push origin main

# 3. Create first alpha release
git tag v0.1.0a1
git push origin v0.1.0a1

# 4. Watch GitHub Actions magic! 🎉
```

In ~3 minutes, users can install with:
```bash
pip install brokle==0.1.0a1
```

## 🔧 Version Strategy

### Alpha Testing (Now)
```bash
git tag v0.1.0a1  # First alpha
git tag v0.1.0a2  # Second alpha (bug fixes)
```

### Beta Testing (Next Week)
```bash
git tag v0.1.0b1  # First beta
git tag v0.1.0b2  # Second beta
```

### Stable Release (Later)
```bash
git tag v0.1.0    # First stable
git tag v0.2.0    # New features
git tag v1.0.0    # Production ready
```

## 🛠️ Customization Options

### Change Version Automatically
The workflows automatically update `pyproject.toml` version from git tags.

### Custom Release Notes
The workflow uses GitHub's auto-generation feature, which automatically creates changelogs from:
- Pull request titles and descriptions
- Commit messages since the last release
- Issue references and mentions

You can customize this by editing the release body in `release.yml` or manually editing releases after creation.

### Different Python Versions
Edit the matrix in `ci.yml`:
```yaml
matrix:
  python-version: ["3.9", "3.10", "3.11", "3.12"]  # Add/remove versions
```

### Integration with Backend
Uncomment the backend testing section in `integration-test.yml` when ready.

## 🎯 What Happens Next

### After First Push
1. **CI runs** → Tests pass → Code quality verified
2. **Build succeeds** → Package ready

### After First Tag
1. **Release created** → GitHub release with assets
2. **PyPI published** → `pip install brokle==0.1.0a1` works
3. **Users can install** → Immediate feedback loop

### Every Future Push
1. **Automatic testing** → Catch issues early
2. **Build verification** → Always ready to release
3. **Security scanning** → Stay secure

## 🎉 Benefits

✅ **Zero-effort publishing** - Just push a tag
✅ **Quality assurance** - Automated testing
✅ **Security** - Automated security scans
✅ **Professional** - Proper versioning and releases
✅ **Fast feedback** - Know immediately if something breaks
✅ **Multiple environments** - Test PyPI then production

## 🚨 Important Notes

1. **Test first:** Always test on Test PyPI before production
2. **Semantic versioning:** Use proper version tags (v0.1.0a1)
3. **Secrets security:** Never commit API tokens
4. **Branch protection:** Consider protecting main branch
5. **Review PRs:** Set up required reviews for quality

Your SDK will be **professionally published** with full CI/CD! 🚀