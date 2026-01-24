from oai_utils.agent import AgentsSDKModel, AgentWrapper
from pydantic import BaseModel

from curriculum.database import Topic


class TopicJudgeResponse(BaseModel):
    useful: bool
    description: str


TOPIC_JUDGE_PROMPT_TEMPLATE = """You are an expert technical educator for the `{library_name}` library.
Your task is to judge if a proposed learning topic is useful for a library USER (someone who wants to USE the library in their projects).

<LIBRARY_SUMMARY>
{library_summary}
</LIBRARY_SUMMARY>


<CRITERIA>
1. **Useful for Users**: Is this something a library user needs to know to be productive?
2. **Not Implementation Detail**: Filter out topics that are purely about how the library is implemented internally (e.g., private structs, internal helper functions) unless they are strictly necessary for the user to understand the public API.
3. **Public API Focus**: Topics should focus on usage patterns, public types, and official features.
</CRITERIA>

Output your judgment in the specified structured format:
- `useful`: A boolean flag.
- `description`: A brief explanation of why this topic is or isn't useful for a library user.
"""


async def judge_topic_usefulness(
    model: AgentsSDKModel, topic: Topic, library_summary: str, library_name: str
) -> TopicJudgeResponse:
    # 1. GPU/CUDA filter (direct skip)
    gpu_keywords = ["gpu", "cuda", "vulkan", "metal", "opencl", "tensorrt"]
    text_to_check = (topic.title + " " + topic.description).lower()
    if any(kw in text_to_check for kw in gpu_keywords):
        return TopicJudgeResponse(
            useful=False,
            description="Topic requires GPU/accelerator which is currently not supported in our environment.",
        )

    # 2. Model-based usefulness filter
    prompt = TOPIC_JUDGE_PROMPT_TEMPLATE.format(
        library_name=library_name,
        library_summary=library_summary,
    )

    agent = AgentWrapper[TopicJudgeResponse].create(
        name="Topic Judge",
        model=model,
        instructions=prompt,
        output_type=TopicJudgeResponse,
    )

    try:
        response = await agent.run(
            f"""\
<PROPOSED_TOPIC>
Title: {topic.title}
APIs: {topic.related_apis}
Description: {topic.description}
</PROPOSED_TOPIC>
            """
        )
        return response.final_output()
    except Exception as e:
        print(f"Error during topic judgment for {topic.id}: {e}")
        # Default to True if judgment fails, to be safe (or False if you prefer strict filtering)
        return TopicJudgeResponse(
            useful=True,
            description=f"Judgment failed due to error: {e}. Defaulting to useful.",
        )
