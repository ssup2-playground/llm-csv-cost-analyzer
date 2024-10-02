import pandas as pd

from langchain.agents.agent_types import AgentType
from langchain_aws import ChatBedrock
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langfuse.callback import CallbackHandler

from prompts import PROMPTS_AWS_COSTS_BY_SERVICE
 
## Variables
MODEL_CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_CLAUDE_3_HAIKU = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"

LANGFUSE_TRACE_NAME = "with-meta-data"

CSV_AWS_COST = "./csv_examples/aws_costs_202309_202408_daily_by_service.csv"

SYSTEM_PROMPT = """
1. Task Context
너는 FinOps 조직의 AWS 비용 분석을 돕는 AI Bot이야. 매일 발생하는 AWS 비용을 살펴보고 FinOps 조직이 놓칠 수 있는 중요한 정보들을 FinOps 팀에게 전달 해야되.

2. Data Description
- 비용 데이터는 일별로 AWS 서비스당 비용을 저장하고 있어.
- 비용 데이터는 2023년 8월부터 2024년 9월까지의 비용을 저장하고 있어.
- 비용 데이터의 첫번째 행은 스키마 정보가 포함되어 있어. 첫 번째 행은 비용 날짜, 마지막 행은 해당 날짜의 총 비용, 그리고 그 사이에는 AWS 서비스별 해당 날짜의 비용이 저장되어 있어.
"""

## Init agent
df_aws_cost_202309_202408 = pd.read_csv(CSV_AWS_COST)

llm = ChatBedrock(
    region_name="us-west-2",
    provider="anthropic",
    model_id=MODEL_CLAUDE_3_5_SONNET,
)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df_aws_cost_202309_202408,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prefix=SYSTEM_PROMPT,
    allow_dangerous_code=True
)

## Evaluate agent with langfuse
langfuse_handler = CallbackHandler()

for index, prompts in enumerate(PROMPTS_AWS_COSTS_BY_SERVICE):
    prompt_input = prompts[0]
    expected_output = prompts[1]
    result = agent.invoke(
        input=prompt_input,
        config={
            "callbacks": [langfuse_handler],
            "run_name": LANGFUSE_TRACE_NAME + str(index), 
        }
    )
    
    print(f"{expected_output in result["output"]} / {result["input"]} / {result["output"]}")
