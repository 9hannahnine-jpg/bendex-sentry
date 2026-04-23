# Changelog

All notable changes to Arc Sentry are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-04-23

### Added
- Dual license: open source under AGPL-3.0, commercial license available from Bendex Geometry LLC
- `COMMERCIAL-LICENSE.md` explaining which license applies to which use case
- "What this is / what this isn't" framing in README, including explicit out-of-distribution behavior disclosure
- "Who this is for" section in README

### Changed
- README rewritten with honest benchmark numbers (92% detection / 0% FPR on a 130-prompt calibrated SaaS deployment benchmark) replacing previous 100%/0% claims
- `pyproject.toml` description updated to match README
- Corrected Crescendo result from Turn 3 to Turn 2 (matching actual test log)
- Research/theory material moved to a single reference at the bottom of the README with a link to bendexgeometry.com/theory
- Version strings aligned to 3.2.0 across `pyproject.toml`, `arc_sentry/__init__.py`, and `arc_sentry.core.pipeline.ArcSentryV3.VERSION`
- `include` pattern in `pyproject.toml` tightened from `arc_sentry*` to `arc_sentry`

### Removed
- Unverified Garak 192/192 claim removed from README pending probe-set verification
- Arc Gate deployment artifacts (`Procfile`, `railway.toml`) removed from the Arc Sentry repo
- `bendex_deprecated/` folder removed from main branch
- `arc_sentry_v2/` removed from main branch; preserved on `legacy-v2` branch for history

## [3.1.1] - 2026 (pre-release)

### Changed
- Package description updated, repository URL corrected

## [3.1.0] - 2026 (pre-release)

### Changed
- Renamed package module from `arc_sentry_v2` to `arc_sentry` with cleaned imports
- Colab quickstart notebook added

## [3.0.0] - 2026 (pre-release)

### Added
- Mean-pooled Fisher-Rao detection method
- Session D(t) monitor for multi-turn (Crescendo-style) attack detection
- Phrase-check first layer for low-latency filtering of obvious injection patterns
