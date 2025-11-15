# Release Changelog Template

Use this template when preparing a new release. Copy the appropriate section to CHANGELOG.md and fill in the details.

## Template for New Release

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature or functionality that has been added
- Example: Added support for streaming responses in OpenAI wrapper
- Example: Added new `trace_id` parameter to observe decorator

### Changed
- Changes to existing functionality
- Example: Updated OpenTelemetry SDK to v1.21.0
- Example: Improved error messages for authentication failures

### Deprecated
- Features that are still available but will be removed in future versions
- Example: Deprecated `old_method()` in favor of `new_method()` (will be removed in v2.0.0)
- Include migration guide if applicable

### Removed
- Features that have been removed
- Example: Removed deprecated `legacy_auth()` function (deprecated since v0.2.0)

### Fixed
- Bug fixes
- Example: Fixed memory leak in batch span processor
- Example: Fixed issue where environment tags were not properly validated

### Security
- Security-related changes or fixes
- Example: Updated dependencies to address CVE-XXXX-XXXX
- Example: Added input sanitization for user-provided metadata

[X.Y.Z]: https://github.com/brokle-ai/brokle-python/releases/tag/vX.Y.Z
```

---

## Guidelines

### Version Numbering (Semantic Versioning)

- **Major (X.0.0)**: Breaking changes, incompatible API changes
- **Minor (0.Y.0)**: New features, backward compatible
- **Patch (0.0.Z)**: Bug fixes, backward compatible

### Pre-release Versions

- **Alpha (X.Y.Z-alpha.N)**: Early testing, unstable, may have bugs
- **Beta (X.Y.Z-beta.N)**: Feature complete, testing for bugs
- **RC (X.Y.Z-rc.N)**: Release candidate, production-ready candidate

### Writing Good Changelog Entries

**Do**:
- ✅ Start with a verb (Added, Fixed, Changed, etc.)
- ✅ Be specific and concise
- ✅ Include issue/PR numbers when relevant (#123)
- ✅ Explain the "why" for breaking changes
- ✅ Group related changes together

**Don't**:
- ❌ Include internal refactoring (unless it affects users)
- ❌ Use vague descriptions ("various improvements")
- ❌ Forget to credit contributors (when applicable)
- ❌ Mix different types of changes in one section

### Examples

**Good**:
```markdown
### Added
- Added `async` support to OpenAI wrapper for better performance (#123)
- Added `timeout` parameter to all HTTP requests (default: 30s)
```

**Bad**:
```markdown
### Added
- Improved things
- Fixed stuff
```

---

## Pre-release Changelog Example

For alpha/beta/rc releases, mark them clearly:

```markdown
## [0.3.0-beta.1] - 2024-11-20

### Added
- **[BETA]** Experimental streaming support for Anthropic models
- **[BETA]** New cost tracking API (subject to change)

### Known Issues
- Streaming may have latency spikes under heavy load
- Cost tracking accuracy is ~95% (improving in beta.2)

### Breaking Changes from v0.2.x
- Removed deprecated `legacy_trace()` method
- Changed `observe()` decorator signature (added `capture_output` parameter)

**Migration Guide**: See [MIGRATION_v0.3.md](docs/MIGRATION_v0.3.md)

[0.3.0-beta.1]: https://github.com/brokle-ai/brokle-python/releases/tag/v0.3.0-beta.1
```

---

## Checklist Before Release

- [ ] All changes documented in appropriate sections
- [ ] Version number follows semantic versioning
- [ ] Date is correct (YYYY-MM-DD format)
- [ ] Version link added at bottom of CHANGELOG.md
- [ ] Breaking changes clearly explained with migration guide
- [ ] Related issues/PRs referenced (#123)
- [ ] Spelling and grammar checked
- [ ] Preview renders correctly on GitHub
