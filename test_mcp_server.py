import os
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# INPUTS - TODO: Update these
DATABRICKS_CLI_PROFILE = os.getenv("DATABRICKS_CLI_PROFILE")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")

workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
host = workspace_client.config.host
# mcp_server_url = f"{host}/api/2.0/mcp/functions/system/ai"
mcp_server_url = f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"


# This snippet below uses the Unity Catalog functions MCP server to expose built-in
# AI tools under `system.ai`, like the `system.ai.python_exec` code interpreter tool
def test_connect_to_server():
    mcp_client = DatabricksMCPClient(
        server_url=mcp_server_url, workspace_client=workspace_client
    )
    tools = mcp_client.list_tools()

    print(
        f"Discovered tools {[t.name for t in tools]} "
        f"from MCP server {mcp_server_url}"
    )


if __name__ == "__main__":
    test_connect_to_server()
