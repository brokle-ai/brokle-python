# Publishing Brokle Python SDK

## Quick Start - Publish to Test PyPI

### 1. Setup PyPI Accounts

1. Create accounts:
   - Test PyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. Generate API tokens:
   - Test PyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. Configure credentials:
```bash
# Create ~/.pypirc file
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

### 2. Publish to Test PyPI (Alpha Testing)

```bash
cd sdk/python

# Install build tools
pip install build twine

# Build the package
make build

# Publish to Test PyPI
make publish-test

# Or manually:
# python -m twine upload --repository testpypi dist/*
```

### 3. Test Installation

```bash
# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ brokle

# Test it works
python -c "from brokle import configure; print('✅ SDK imported successfully!')"
```

### 4. Publish to Main PyPI

Once tested on Test PyPI:

```bash
# Publish to main PyPI
make publish

# Or manually:
# python -m twine upload dist/*
```

## Version Management Strategy

### Alpha/Beta Versions (Now)
- `0.1.0a1` - Alpha 1
- `0.1.0a2` - Alpha 2
- `0.1.0b1` - Beta 1
- `0.1.0b2` - Beta 2

### Release Versions (Later)
- `0.1.0` - First stable release
- `0.1.1` - Bug fixes
- `0.2.0` - New features
- `1.0.0` - Production ready

### Update Version

Edit `pyproject.toml`:
```toml
[project]
name = "brokle"
version = "0.1.0a1"  # Alpha version
```

## Publishing Checklist

Before publishing:

- [ ] Tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] Type checking passes: `make type-check`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version number is correct
- [ ] License is included
- [ ] README.md has installation instructions

## Continuous Integration

For automated publishing, add to `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Post-Publication

After publishing:

1. **Update documentation:**
   - Add installation instructions
   - Update examples to use `pip install brokle`

2. **Announce:**
   - GitHub release notes
   - Documentation updates
   - Community announcements

3. **Monitor:**
   - Download statistics
   - User feedback
   - Bug reports

## Common Issues

### "Package already exists"
- Check if name is taken: https://pypi.org/project/brokle/
- Use different name if needed: `brokle-platform`, `brokle-sdk`

### "Authentication failed"
- Check API token is correct
- Ensure token has upload permissions
- Verify ~/.pypirc configuration

### "Invalid distribution"
- Run `make clean build` to rebuild
- Check pyproject.toml syntax
- Ensure all required files are included

## Success! 🎉

Once published, users can install with:

```bash
# From Test PyPI (alpha)
pip install -i https://test.pypi.org/simple/ brokle

# From main PyPI (beta/stable)
pip install brokle
```