
To run the `cli.py` script and use it to manage the `server.py`, follow these steps:

1. **Install uv and dependences**
   Assumed you installed dependencies already.

2. **Run the CLI Script**:
   You can execute the `cli.py` script directly using Python. For example:
   ```bash
   uv run src/openai_responses_server/cli.py <command>
   ```
   Replace `<command>` with one of the available commands (`start`, `configure`, or `help`).

3. **Available Commands**:
   - `start`: Starts the FastAPI server defined in `server.py`.
   - `configure`: Allows you to configure server settings like host, port, API URLs, and API key.
   - `help`: Displays help information about the CLI.

4. **Example Usage**:
   - To start the server:
     ```bash
     python src/openai_responses_server/cli.py start
     ```
   - To configure the server:
     ```bash
     python src/openai_responses_server/cli.py configure
     ```
   - To display help:
     ```bash
     python src/openai_responses_server/cli.py help
     ```

5. **Make the Script Executable (Optional)**:
   If you want to run the script without explicitly calling Python, you can make it executable:
   ```bash
   chmod +x src/openai_responses_server/cli.py
   ```
   Then, run it directly:
   ```bash
   ./src/openai_responses_server/cli.py <command>
   ```

Let me know if you need further assistance!