import argparse

parser = argparse.ArgumentParser(description="AI Gun - Rick and Morty Edition")
parser.add_argument("image_path", type=str, help="Optional path to the input image.", nargs="?")
parser.add_argument(
    "--use_real_camera",
    action="store_true",
    help="Use a real camera feed instead of a static image.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug logging.",
)
parser.add_argument(
    "--audio_output",
    action="store_true",
    help="Enable audio output.",
)
parser.add_argument(
    "--model",
    type=str,
    help="Specify the model to use for the agent.",
)
args = parser.parse_args()

import logging

import dotenv
from langchain.agents import create_agent

import rnm.config as C
from rnm.tools.boundingBox import main as bounding_box_tool
from rnm.tools.visual_io import camera, display_image

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)


def process_args(args):
    print(f"Arguments: {args}")
    exit()
    C.USE_REAL_CAMERA = args.use_real_camera
    C.DEBUG = args.debug
    C.AUDIO_OUTPUT = args.audio_output
    if args.model:
        C.model = args.model


def main():
    process_args(args)

    logging.info("Starting AI Gun - Rick and Morty Edition")
    SYSTEM_PROMPT = """
You are helpful Super AI Gun agent that plans before acting. Use the tools available to you to accomplish the user's request.
Decide what information you need and which tools to use before responding.
Execute tools as necessary, then provide the final answer by displaying the image.

Special and important instructions:
- Follow the instructions in the user message carefully, and do not assume anything without being absolutely sure.
- ** Always display the image once the camera tools is used, so that the user can see what was captured. (always call the display_image tool after using the camera tool) **
- If any tool returns an image path, always display the image to the user (even if the user did not explicitly ask for it).
"""
    agent = create_agent(
        # model="ollama:qwen3.5:4b",
        # model="ollama:granite4.1:3b",
        model=f"ollama:{C.model}",
        # model="ollama:qwen3:4b",
        # model="ollama:ornith:latest",
        tools=[camera, display_image, bounding_box_tool],
        # middleware=[TodoListMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        debug=C.DEBUG,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    # "content": "Take a picture using the camera, find the alien faces, and shoot them.",
                    "content": "Take a picture using the camera, find the alien faces, and shoot them.",
                }
            ]
        },
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
