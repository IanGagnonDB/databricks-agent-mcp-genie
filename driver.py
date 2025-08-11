import os
import json
import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksGenieSpace,
    DatabricksTable,
)
from databricks import agents

# region INPUTS
DATABRICKS_CLI_PROFILE = os.environ["DATABRICKS_CLI_PROFILE"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]

LLM_ENDPOINT_NAME = os.environ["LLM_ENDPOINT_NAME"]
GENIE_SPACE_ID = os.environ["GENIE_SPACE_ID"]
TABLE_NAMES = json.loads(os.environ["TABLE_NAMES"])

CATALOG_NAME = os.environ["CATALOG_NAME"]
SCHEMA_NAME = os.environ["SCHEMA_NAME"]
MODEL_NAME = os.environ["MODEL_NAME"]
# endregion INPUTS


# region 0 - SETUP
print("\n" + "-" * 50)
print("Starting Step 0 (Setup)...")
current_dir = os.path.dirname(os.path.abspath(__file__))
mcp_agent_path = os.path.join(current_dir, "mcp_agent.py")

os.environ["DATABRICKS_CONFIG_PROFILE"] = DATABRICKS_CLI_PROFILE
# Set tracking URI to Databricks so models go to your workspace
mlflow.set_tracking_uri(f"databricks://{DATABRICKS_CLI_PROFILE}")
mlflow.set_experiment(EXPERIMENT_NAME)
print("Step 0 (Setup) completed.")
# endregion 0 - SETUP


# region 1 - LOG MODEL
print("\n" + "-" * 50)
print("Starting Step 1 (Log Model)...")
# TODO: Add ALL the resources used by the model to allow system authentication and prevent authorization errors
# ref: https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent#-specify-resources-for-automatic-authentication-passthrough-system-authentication
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    *[DatabricksTable(table_name=table_name) for table_name in TABLE_NAMES],
]

input_example = {
    "input": [{"role": "user", "content": "Who created the most pipeline?"}]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name=MODEL_NAME,
        python_model=mcp_agent_path,
        resources=resources,
        input_example=input_example,
    )

print(f"MLflow Run: {logged_agent_info.run_id}")
print(f"Model URI: {logged_agent_info.model_uri}")
print("Step 1 (Log Model) completed.")
# endregion 1 - LOG MODEL


# region 2 - REGISTER MODEL
print("\n" + "-" * 50)
print("Starting Step 2 (Register Model)...")
mlflow.set_registry_uri("databricks-uc")
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=model_name
)
print("Step 2 (Register Model) completed.")
# endregion 2 - REGISTER MODEL


# region 3 - DEPLOY AGENT
print("\n" + "-" * 50)
print("Starting Step 3 (Deploy Agent)...")
deployment = agents.deploy(
    model_name=model_name,
    model_version=model_info.version,
)
print("Agent query endpoint:", deployment.query_endpoint)

# OPTIONAL: Wait for deployment to complete
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(
    deployment.endpoint_name
)
print(f"Final deployment status: {endpoint.state.config_update}")
print(f"Final config version: {endpoint.config.config_version}")
print("Step 3 (Deploy Agent) completed.")
# endregion 3 - DEPLOY AGENT
