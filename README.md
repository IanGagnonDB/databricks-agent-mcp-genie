# Databricks MCP Agent Example

This project demonstrates how to create an Agent in Databricks using Managed MCP (Model Context Protocol) Server for Genie. The agent integrates with Databricks Genie spaces and other MCP-enabled services to provide intelligent data querying capabilities.

## Overview

The MCP Agent leverages Databricks' managed MCP servers to create a conversational AI agent that can:
- Query data through Genie spaces
- Access Unity Catalog functions
- Interact with various Databricks resources
- Provide intelligent responses using LLM endpoints

## Project Structure

- `driver.py` - Main orchestration script that logs, registers, and deploys the agent
- `mcp_agent.py` - Core agent implementation using MLflow's ResponsesAgent framework
- `test_mcp_server.py` - Test script to verify MCP server connectivity
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Prerequisites

### Python Environment Setup
A Python environment must be properly set up using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Required Environment Variables
The following environment variables must be defined before running the project:

```bash
# Databricks Configuration
export DATABRICKS_CLI_PROFILE="your-databricks-profile"
export EXPERIMENT_NAME="your-mlflow-experiment-name"

# Model and Endpoint Configuration
export LLM_ENDPOINT_NAME="your-llm-endpoint-name"
export GENIE_SPACE_ID="your-genie-space-id"

# Unity Catalog Configuration
export CATALOG_NAME="your-catalog-name"
export SCHEMA_NAME="your-schema-name"
export MODEL_NAME="your-model-name"

# Table Access (JSON array format)
export TABLE_NAMES='["table1", "table2", "table3"]'
```

### Databricks Setup
1. **Databricks CLI**: Ensure the Databricks CLI is installed and configured with appropriate authentication
2. **Workspace Access**: Your profile must have access to the target Databricks workspace
3. **Genie Space**: A configured Genie space with the specified `GENIE_SPACE_ID`
4. **LLM Endpoint**: Access to a serving endpoint for the LLM (e.g., Claude, GPT models)
5. **Unity Catalog**: Proper permissions to the specified catalog and schema

## Usage

### 1. Test MCP Server Connectivity
First, verify that your MCP server connection is working:

```bash
python test_mcp_server.py
```

This will list available tools from your configured MCP server.

### 2. Deploy the Agent
Run the main driver script to log, register, and deploy your agent:

```bash
python driver.py
```

This script will:
1. Log the agent model to MLflow
2. Register the model in Unity Catalog
3. Deploy the agent as a serving endpoint


## Configuration

### MCP Server URLs
The agent can connect to multiple MCP servers. Configure them in `mcp_agent.py`:

```python
MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}",
    # Add more managed MCP servers as needed
]

CUSTOM_MCP_SERVER_URLS = [
    # Add custom MCP servers hosted on Databricks Apps
]
```

### System Prompt
Customize the agent's behavior by modifying the `SYSTEM_PROMPT` in `mcp_agent.py`:

```python
SYSTEM_PROMPT = "You are a helpful assistant specialized in data analysis."
```

## Architecture

The agent follows this architecture:
1. **Request Processing**: Receives user queries through the ResponsesAgent interface
2. **Tool Discovery**: Dynamically discovers available tools from configured MCP servers
3. **LLM Integration**: Uses Databricks serving endpoints for language model inference
4. **Tool Execution**: Executes MCP tools (e.g., Genie queries) based on LLM decisions
5. **Response Generation**: Provides intelligent responses based on tool outputs

## Security and Authentication

- The agent uses Databricks system authentication for resource access
- All specified resources in the `driver.py` are automatically granted access
- MCP servers authenticate through the Databricks workspace client

## Troubleshooting

### Common Issues
1. **Authentication Errors**: Verify your `DATABRICKS_CLI_PROFILE` is correctly configured
2. **Resource Access**: Ensure all tables and endpoints specified in environment variables are accessible
3. **MCP Server Connectivity**: Use `test_mcp_server.py` to verify server connections
4. **Environment Variables**: Double-check that all required environment variables are set

### Debugging
- Check MLflow experiment logs for model registration issues
- Verify Genie space accessibility through the Databricks workspace UI
- Test individual MCP tools using the `DatabricksMCPClient` directly

## Example Queries

Once deployed, your agent can handle queries like:
- "Who created the most pipelines?"
- "Show me the latest data quality metrics"
- "What are the top performing models this month?"

The agent will automatically determine which tools to use and provide intelligent responses based on your data.
