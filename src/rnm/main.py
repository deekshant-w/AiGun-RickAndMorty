import argparse
import logging

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from rnm.tools import visual_io
from rnm.tools.boundingBox import main as bounding_box_tool

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="AI Gun - Rick and Morty Edition")
parser.add_argument(
    "--use_real_camera",
    action="store_true",
    help="Use a real camera feed instead of a static image.",
)


def process_args(args):
    visual_io.USE_REAL_CAMERA = args.use_real_camera


def main():
    process_args(parser.parse_args())

    logging.info("Starting AI Gun - Rick and Morty Edition")
    SYSTEM_PROMPT = """
You are helpful Super AI Gun agent that plans before acting.
Decide what information you need and which tools to use.
Execute tools as necessary, then provide the final answer by displaying the image.

Special and important instructions:
- Follow the instructions in the user message carefully, and do not assume anything without being absolutely sure.
- ** Always display the image once the camera is used, so that the user can see what was captured. **
- If any tool returns an image path, always display the image to the user.
"""
    agent = create_agent(
        model="ollama:qwen3.5:latest",
        tools=[visual_io.camera, visual_io.display_image, bounding_box_tool],
        middleware=[TodoListMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        debug=True,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Take a picture using the camera, find the alien faces, and shoot them.",
                }
            ]
        },
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
