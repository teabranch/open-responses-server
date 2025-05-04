# Security Scanning Setup

This project uses multiple security scanning tools to ensure code quality and security. Below is information about the setup and how to use these tools.

## Automated Security Scans

The following security checks are automatically run on each push to main and on each pull request:

### 1. Bandit

[Bandit](https://bandit.readthedocs.io/) is a tool designed to find common security issues in Python code.

- Results are available as GitHub workflow artifacts
- Configuration is in the `.github/workflows/security-scan.yml` file

### 2. Pylint

[Pylint](https://www.pylint.org/) is a static code analysis tool that looks for programming errors, helps enforce coding standards, and can detect some security issues.

- Configuration is stored in `.pylintrc`
- Results are available as GitHub workflow artifacts

### 3. SonarCloud

[SonarCloud](https://sonarcloud.io/) provides a comprehensive code quality and security analysis for cloud repositories.

- Results are available in the SonarCloud dashboard after signing in
- Configuration is stored in `sonar-project.properties`

#### Setting up SonarCloud

1. Go to [SonarCloud](https://sonarcloud.io/) and log in with your GitHub account
2. Create a new organization or use an existing one
3. Add your repository to SonarCloud
4. Generate a token in SonarCloud: Account > Security > Generate Token
5. Add the token as a secret named `SONAR_TOKEN` in your GitHub repository settings
6. The GitHub workflows are already configured to use SonarCloud with your organization

## Local Development Setup

### Pre-commit Hooks

To ensure code quality and security before committing, you can use the pre-commit hooks:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hooks:
```bash
pre-commit install
```

3. The hooks will now run automatically on each commit

### Manual Security Scanning

You can also run the security tools manually:

#### Bandit
```bash
pip install bandit
bandit -r src/
```

#### Pylint
```bash
pip install pylint
pylint src/
```

## Security Best Practices

When contributing to this project, please follow these security best practices:

1. **Input Validation**: Always validate and sanitize user inputs
2. **Avoid Hardcoded Secrets**: Never commit secrets, API keys, or credentials
3. **Use Parameterized Queries**: Prevent SQL injection by using parameterized queries
4. **Secure Dependencies**: Regularly update dependencies to mitigate security vulnerabilities
5. **Error Handling**: Implement proper error handling to avoid exposing sensitive information
6. **Secure HTTP Headers**: Set appropriate security headers for web applications
7. **Use HTTPS**: Always use HTTPS for API requests
8. **Authentication and Authorization**: Implement proper authentication and authorization checks

## Reporting Security Issues

If you discover a security vulnerability, please do NOT open an issue. Email [security contact email] instead. 