import json
import uuid
import asyncio
from typing import Any, Callable, List
from pydantic import BaseModel

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# INPUTS - TODO: Update these
DATABRICKS_CLI_PROFILE = "field-eng-east"
GENIE_SPACE_ID = "01f05dfe18a8101da6e90846a5d13a45"

# 1) CONFIGURE YOUR ENDPOINTS/PROFILE
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
SYSTEM_PROMPT = "You are a helpful assistant."
assert (
    DATABRICKS_CLI_PROFILE != "YOUR_DATABRICKS_CLI_PROFILE"
), "Set DATABRICKS_CLI_PROFILE to the Databricks CLI profile name you specified when configuring authentication to the workspace"
workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
host = workspace_client.config.host
# Add more MCP server URLs here if desired, e.g
# f"{host}/api/2.0/mcp/vector-search/prod/billing"
# to include vector search indexes under the prod.billing schema, or
# f"{host}/api/2.0/mcp/genie/<genie_space_id>"
# to include a Genie space
MANAGED_MCP_SERVER_URLS = [
#  f"{host}/api/2.0/mcp/functions/system/ai",
    f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}",
]
# Add Custom MCP Servers hosted on Databricks Apps
CUSTOM_MCP_SERVER_URLS = []



# 2) HELPER: convert between ResponsesAgent “message dict” and ChatCompletions format
def _to_chat_messages(msg: dict[str, Any]) -> List[dict]:
    """
    Take a single ResponsesAgent-style dict and turn it into one or more
    ChatCompletions-compatible dict entries.
    """
    msg_type = msg.get("type")
    if msg_type == "function_call":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": msg["call_id"],
                        "type": "function",
                        "function": {
                            "name": msg["name"],
                            "arguments": msg["arguments"],
                        },
                    }
                ],
            }
        ]
    elif msg_type == "message" and isinstance(msg["content"], list):
        return [
            {
                "role": "assistant" if msg["role"] == "assistant" else msg["role"],
                "content": content["text"],
            }
            for content in msg["content"]
        ]
    elif msg_type == "function_call_output":
        return [
            {
                "role": "tool",
                "content": msg["output"],
                "tool_call_id": msg["tool_call_id"],
            }
        ]
    else:
        # fallback for plain {"role": ..., "content": "..."} or similar
        return [
            {
                k: v
                for k, v in msg.items()
                if k in ("role", "content", "name", "tool_calls", "tool_call_id")
            }
        ]


# 3) “MCP SESSION” + TOOL‐INVOCATION LOGIC
def _make_exec_fn(
    server_url: str, tool_name: str, ws: WorkspaceClient
) -> Callable[..., str]:
    def exec_fn(**kwargs):
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
        response = mcp_client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable


def _fetch_tool_infos(ws: WorkspaceClient, server_url: str) -> List[ToolInfo]:
    print(f"Listing tools from MCP server {server_url}")
    infos: List[ToolInfo] = []
    mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
    mcp_tools = mcp_client.list_tools()
    for t in mcp_tools:
        schema = t.inputSchema.copy()
        if "properties" not in schema:
            schema["properties"] = {}
        spec = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema,
            },
        }
        infos.append(
            ToolInfo(
                name=t.name, spec=spec, exec_fn=_make_exec_fn(server_url, t.name, ws)
            )
        )
    return infos


# 4) “SINGLE‐TURN” AGENT CLASS
class SingleTurnMCPAgent(ResponsesAgent):
    def _call_llm(self, history: List[dict], ws: WorkspaceClient, tool_infos):
        """
        Send current history → LLM, returning the raw response dict.
        """
        client = ws.serving_endpoints.get_open_ai_client()
        flat_msgs = []
        for msg in history:
            flat_msgs.extend(_to_chat_messages(msg))
        return client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=flat_msgs,
            tools=[ti.spec for ti in tool_infos],
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        ws = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)

        # 1) build initial history: system + user
        history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for inp in request.input:
            history.append(inp.model_dump())

        # 2) call LLM once
        tool_infos = [
            tool_info
            for mcp_server_url in (MANAGED_MCP_SERVER_URLS + CUSTOM_MCP_SERVER_URLS)
            for tool_info in _fetch_tool_infos(ws, mcp_server_url)
        ]
        tools_dict = {tool_info.name: tool_info for tool_info in tool_infos}
        llm_resp = self._call_llm(history, ws, tool_infos)
        raw_choice = llm_resp.choices[0].message.to_dict()
        raw_choice["id"] = uuid.uuid4().hex
        history.append(raw_choice)

        tool_calls = raw_choice.get("tool_calls") or []
        if tool_calls:
            # (we only support a single tool in this “single‐turn” example)
            fc = tool_calls[0]
            name = fc["function"]["name"]
            args = json.loads(fc["function"]["arguments"])
            try:
                tool_info = tools_dict[name]
                result = tool_info.exec_fn(**args)
            except Exception as e:
                result = f"Error invoking {name}: {e}"

            # 4) append the “tool” output
            history.append(
                {
                    "type": "function_call_output",
                    "role": "tool",
                    "id": uuid.uuid4().hex,
                    "tool_call_id": fc["id"],
                    "output": result,
                }
            )

            # 5) call LLM a second time and treat that reply as final
            followup = (
                self._call_llm(history, ws, tool_infos=[]).choices[0].message.to_dict()
            )
            followup["id"] = uuid.uuid4().hex

            assistant_text = followup.get("content", "")
            return ResponsesAgentResponse(
                output=[
                    {
                        "id": uuid.uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": assistant_text}],
                    }
                ],
                custom_outputs=request.custom_inputs,
            )

        # 6) if no tool_calls at all, return the assistant’s original reply
        assistant_text = raw_choice.get("content", "")
        return ResponsesAgentResponse(
            output=[
                {
                    "id": uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant_text}],
                }
            ],
            custom_outputs=request.custom_inputs,
        )


mlflow.models.set_model(SingleTurnMCPAgent())

if __name__ == "__main__":
    req = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Who created the most pipeline?"}]
    )
    resp = SingleTurnMCPAgent().predict(req)
    for item in resp.output:
        print(item)
