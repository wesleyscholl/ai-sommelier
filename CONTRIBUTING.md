# Contributing to AI Sommelier

Thank you for your interest in contributing to AI Sommelier! This document provides guidelines for contributing to this production wine recommendation system.

## 🍷 Project Overview

AI Sommelier is a live production application deployed on Streamlit Cloud that provides AI-powered wine recommendations. Contributions should maintain the high quality and user experience standards of the production system.

## 🤝 Ways to Contribute

### 1. Wine Data & Knowledge
- **Expand Wine Database** - Add wine profiles, tasting notes, food pairings
- **Regional Expertise** - Contribute knowledge about specific wine regions
- **Vintage Information** - Add vintage quality ratings and aging potential
- **Food Pairing Data** - Expand the food pairing recommendation engine

### 2. Features & Enhancements
- **Search Improvements** - Better filtering, sorting, and recommendation algorithms
- **UI/UX Enhancements** - Improve the Streamlit interface and user flow
- **Mobile Optimization** - Better mobile experience
- **Accessibility** - WCAG compliance improvements

### 3. AI Model Improvements
- **Prompt Engineering** - Refine AI prompts for better recommendations
- **Model Testing** - Test different LLM models for quality and cost
- **Response Quality** - Improve consistency and accuracy of recommendations
- **Context Understanding** - Better interpretation of user preferences

### 4. Documentation
- **User Guides** - Tutorials for wine enthusiasts
- **API Documentation** - If adding programmatic interfaces
- **Wine Education** - Explanatory content about wine regions, varietals, terminology

### 5. Testing & Quality
- **Unit Tests** - Test coverage for core functionality
- **Integration Tests** - Test AI model interactions
- **User Testing** - Feedback on UX and recommendation quality
- **Performance Testing** - Load testing and optimization

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Streamlit account (for deployment testing)
- OpenAI API key or alternative LLM API
- Basic wine knowledge (helpful but not required!)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/wesleyscholl/ai-sommelier.git
cd ai-sommelier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
streamlit run app.py
```

### Testing Locally

```bash
# Run tests (if test suite exists)
pytest tests/

# Run type checking
mypy app.py

# Run linting
flake8 app.py
black app.py --check

# Test Streamlit app
streamlit run app.py
# Navigate to http://localhost:8501
```

## 📝 Contribution Process

### 1. Find or Create an Issue
- Check existing issues for tasks to work on
- Create new issue for bugs or feature requests
- Get issue assigned before starting work (prevents duplicate effort)

### 2. Fork and Branch
```bash
# Fork repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-sommelier.git
cd ai-sommelier

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bugs: git checkout -b fix/bug-description
```

### 3. Make Changes
- Follow code style guidelines (below)
- Write clear, concise commit messages
- Add tests for new features
- Update documentation as needed

### 4. Test Thoroughly
- Test all functionality locally
- Verify Streamlit UI works correctly
- Check AI responses for quality
- Test edge cases and error handling

### 5. Submit Pull Request
```bash
# Commit changes
git add .
git commit -m "feat: add wine region filter"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

**Pull Request Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How was this tested?

## Screenshots (if UI changes)
Before/after screenshots

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented hard-to-understand areas
- [ ] Updated documentation
- [ ] Added tests
- [ ] All tests pass
- [ ] No new warnings
```

## 🎨 Code Style Guidelines

### Python Style
- **PEP 8** compliance
- **Type hints** for all functions
- **Docstrings** for modules, classes, functions
- **Black** formatter (line length: 88)
- **isort** for import sorting

### Example:
```python
from typing import List, Dict, Optional
import streamlit as st
from openai import OpenAI

def get_wine_recommendations(
    preferences: Dict[str, str],
    budget: Optional[float] = None,
    occasion: str = "casual"
) -> List[Dict[str, str]]:
    """
    Generate wine recommendations based on user preferences.
    
    Args:
        preferences: Dictionary of user preferences (style, region, etc.)
        budget: Optional budget constraint in USD
        occasion: Type of occasion (casual, formal, celebration)
        
    Returns:
        List of wine recommendations with details
        
    Raises:
        ValueError: If preferences are invalid
    """
    # Implementation here
    pass
```

### Streamlit Best Practices
- Use `st.cache_data` for expensive operations
- Minimize API calls with caching
- Provide loading states and progress indicators
- Handle errors gracefully with user-friendly messages
- Optimize for mobile responsiveness

## 🧪 Testing Guidelines

### Unit Tests
```python
def test_wine_recommendation_basic():
    """Test basic wine recommendation generation."""
    prefs = {"style": "red", "body": "full"}
    recommendations = get_wine_recommendations(prefs)
    assert len(recommendations) > 0
    assert all("name" in rec for rec in recommendations)
```

### Integration Tests
```python
def test_openai_integration():
    """Test OpenAI API integration."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = generate_recommendation_prompt("Cabernet Sauvignon")
    assert response is not None
    assert len(response) > 50
```

## 🌍 Community Guidelines

### Be Respectful
- Treat all contributors with respect
- Provide constructive feedback
- Welcome newcomers and help them get started
- Celebrate diverse perspectives and backgrounds

### Communication
- Use clear, professional language
- Provide context in issues and PRs
- Respond to feedback promptly
- Ask questions when unclear

### Quality Standards
- Maintain production-level quality
- Test thoroughly before submitting
- Document your changes
- Consider edge cases and error handling

## 🐛 Reporting Bugs

**Good Bug Report Includes:**
1. **Clear title** - Concise description of issue
2. **Steps to reproduce** - Exact steps to trigger bug
3. **Expected behavior** - What should happen
4. **Actual behavior** - What actually happens
5. **Environment** - OS, Python version, browser
6. **Screenshots** - If applicable
7. **Error logs** - Any error messages or stack traces

## 💡 Suggesting Features

**Good Feature Request Includes:**
1. **Problem statement** - What problem does this solve?
2. **Proposed solution** - How would it work?
3. **Alternatives considered** - Other approaches
4. **Use cases** - When would users need this?
5. **Mockups** - UI mockups if relevant

## 📊 Wine Data Contributions

### Data Format
Wine data should follow this structure:
```json
{
  "name": "Wine Name",
  "producer": "Producer Name",
  "region": "Region",
  "country": "Country",
  "varietal": "Grape Varietal",
  "vintage": 2020,
  "style": "Red/White/Rosé/Sparkling",
  "body": "Light/Medium/Full",
  "tasting_notes": ["cherry", "oak", "vanilla"],
  "food_pairings": ["steak", "lamb", "aged cheese"],
  "price_range": "$15-25",
  "rating": 4.2
}
```

### Data Quality Requirements
- **Accuracy** - Verify information from reliable sources
- **Completeness** - Include all required fields
- **Consistency** - Follow naming conventions
- **Attribution** - Credit data sources
- **Licensing** - Ensure data can be used freely

## 🚀 Deployment

### Production Deployment (Streamlit Cloud)
- Only maintainers can deploy to production
- All PRs must pass CI/CD checks
- Production requires manual review and approval
- Monitor performance after deployment

### Testing Deployment
Contributors can test on personal Streamlit Cloud instances:
1. Fork repository
2. Connect to Streamlit Cloud
3. Deploy your fork for testing
4. Share link in PR for review

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📧 Questions?

- **Issues**: Open an issue for questions
- **Email**: Contact maintainer via GitHub profile
- **Discussions**: Use GitHub Discussions for general questions

---

**Thank you for contributing to AI Sommelier! Together we're making wine discovery more accessible and enjoyable for everyone. 🍷**
