# Changelog

All notable changes to AI Sommelier will be documented in this file.

## [Unreleased]

### Added
- **Comprehensive Test Suite** - Achieved 79% code coverage
  - 55 unit tests covering all major components
  - Test coverage: recommender.py (73%), sommelier.py (84%), utils.py (96%)
  - Test suites for:
    - Recommender initialization, fitting, recommendation, save/load
    - Sommelier explanation generation with Gemini and fallbacks
    - Wine dataset loading and preprocessing utilities
- **Testing Infrastructure** - pytest with coverage reporting
  - HTML coverage reports in htmlcov/
  - Comprehensive edge case testing
  - Mock testing for external API dependencies
- **Quality Assurance** - All tests passing with robust error handling

### Changed
- Enhanced test documentation and organization
- Improved code reliability through extensive testing

## [1.0.0] - 2025-11-04

### Initial Release
- Wine recommendation engine with sentence transformers
- Gemini AI integration for sommelier explanations
- Streamlit web interface
- Price and variety filtering
- Save/load embeddings functionality
