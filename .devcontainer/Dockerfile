FROM mcr.microsoft.com/vscode/devcontainers/cpp:latest

# Install any additional packages here
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the environment for the Milestone Manager extension
RUN mkdir -p /workspace/.vscode-server/extensions \
    && curl -L -o /workspace/.vscode-server/extensions/milestone-manager.zip https://example.com/path/to/milestone-manager.zip \
    && unzip /workspace/.vscode-server/extensions/milestone-manager.zip -d /workspace/.vscode-server/extensions/ \
    && rm /workspace/.vscode-server/extensions/milestone-manager.zip

# Set the working directory
WORKDIR /workspace

# Expose any ports if necessary
# EXPOSE 8080

# Command to run when starting the container
CMD ["sleep", "infinity"]