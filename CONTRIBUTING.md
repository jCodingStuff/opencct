# Contributing to OpenCCT

Thank you for your interest in contributing to OpenCCT! This document provides guidelines and instructions for contributing to the project.

## Development Status

OpenCCT is currently under active development. APIs are subject to change, and we're building out core functionality. Contributions are welcome, but please be aware that significant refactoring may occur.

## Getting Started

### Prerequisites

- Rust (latest stable version recommended)
- Git
- Python 3.7+ (for pre-commit hooks)

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/opencct.git
   cd opencct
   ```

2. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

   This will automatically run `cargo fmt`, `cargo check`, and `cargo clippy` before each commit.

3. **Build the project:**
   ```bash
   cargo build
   ```

4. **Run tests:**
   ```bash
   cargo test
   ```

   To run expensive statistical tests (ignored by default):
   ```bash
   cargo test -- --ignored
   ```

## Code Style

We follow standard Rust conventions:

- **Formatting**: Code is automatically formatted with `rustfmt` via pre-commit hooks
- **Linting**: `clippy` is run with `-D warnings` (all warnings treated as errors)
- **Documentation**: Public items should have doc comments
- **Tests**: New functionality should include tests

The pre-commit hooks will enforce these automatically, but you can run them manually:

```bash
cargo fmt
cargo clippy --all-features -- -D warnings
cargo test
```

## Making Changes

### Branching Strategy

- `main`: Stable releases
- `dev`: Active development branch
- Feature branches: Create from `dev` with descriptive names (e.g., `feature/acd-routing`, `fix/exponential-sampling`)

### Commit Messages

Write clear, descriptive commit messages:

```
Add ACD routing with priority matrix

- Implement priority-based routing algorithm
- Add longest-idle tiebreaker
- Include tests for routing scenarios
```

### Pull Request Process

1. **Create a feature branch from `dev`:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   - Pre-commit hooks will run automatically
   - Ensure all tests pass: `cargo test`

3. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request:**
   - Target the `dev` branch (not `main`)
   - Provide a clear description of the changes
   - Reference any related issues

5. **Code Review:**
   - Address any feedback from reviewers
   - Keep your branch up to date with `dev`

## Testing

### Running Tests

```bash
# Run all tests except expensive statistical tests
cargo test

# Run all tests including statistical tests
cargo test -- --ignored --test-threads=1

# Run tests for a specific module
cargo test distributions::uniform
```

### Writing Tests

- Unit tests: Place in the same file as the code, in a `tests` module
- Integration tests: Place in `tests/` directory
- Statistical tests: Mark expensive tests with `#[ignore]` attribute

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test code
    }

    #[test]
    #[ignore]  // Expensive statistical test
    fn test_distribution_properties() {
        // Statistical validation
    }
}
```

## Areas for Contribution

Current focus areas (check the [issues](https://github.com/jCodingStuff/opencct/issues) for specific tasks):

- **Simulation Engine**: Discrete event simulation, ACD routing
- **Queuing Models**: Erlang-C, M/M/c implementations
- **Optimization**: Staffing optimization algorithms
- **Documentation**: Examples, tutorials, API documentation
- **Testing**: Additional test coverage, benchmarks

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For questions about the codebase, feel free to open a discussion

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing to OpenCCT, you agree that your contributions will be licensed under the MIT or Apache-2.0 licenses (dual-licensed).