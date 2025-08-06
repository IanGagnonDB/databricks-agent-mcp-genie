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
DATABRICKS_CLI_PROFILE = os.getenv("DATABRICKS_CLI_PROFILE")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
TABLE_NAMES = json.loads(os.getenv("TABLE_NAMES"))

CATALOG_NAME = os.getenv("CATALOG_NAME")
SCHEMA_NAME = os.getenv("SCHEMA_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
# endregion INPUTS


# region 0 - SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
mcp_agent_path = os.path.join(current_dir, "mcp_agent.py")

os.environ["DATABRICKS_CONFIG_PROFILE"] = DATABRICKS_CLI_PROFILE
# Set tracking URI to Databricks so models go to your workspace
mlflow.set_tracking_uri(f"databricks://{DATABRICKS_CLI_PROFILE}")
mlflow.set_experiment(EXPERIMENT_NAME)
# endregion 0 - SETUP


# region 1 - LOG MODEL
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
# endregion 1 - LOG MODEL


# region 2 - REGISTER MODEL
mlflow.set_registry_uri("databricks-uc")
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=model_name
)
# endregion 2 - REGISTER MODEL


# region 3 - DEPLOY AGENT
deployment = agents.deploy(
    model_name=model_name,
    model_version=model_info.version,
)
print("Agent query endpoint:", deployment.query_endpoint)
# endregion 3 - DEPLOY AGENT
