# Contributing to Responsible AI Implementation

We love your input! We want to make contributing to Responsible AI Implementation as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/kingogie88/responsible-ai-implementation/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/kingogie88/responsible-ai-implementation/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Follow code style guidelines:
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Add type hints to all functions
- Write docstrings in Google style

4. Run tests:
```bash
pytest tests/
```

5. Check code quality:
```bash
flake8 src/
mypy src/
```

## Documentation

- Update documentation for any new features or changes
- Follow the documentation style guide
- Test documentation examples
- Update the changelog

## License
By contributing, you agree that your contributions will be licensed under its MIT License. 