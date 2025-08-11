import os
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# INPUTS - TODO: Update these
DATABRICKS_CLI_PROFILE = os.environ["DATABRICKS_CLI_PROFILE"]
GENIE_SPACE_ID = os.environ["GENIE_SPACE_ID"]


workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
host = workspace_client.config.host
mcp_server_url = f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"


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
