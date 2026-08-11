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
    "--disable_audio_output",
    action="store_true",
    help="Disable audio output.",
)
parser.add_argument(
    "--model",
    type=str,
    help="Specify the model to use for the agent.",
)
args = parser.parse_args()

import random
import threading
from pathlib import Path

import dotenv
from langchain.agents import create_agent

import rnm.config as C
import rnm.tools.misc_tools as misc_tools
from rnm.tools.audio_io import inputLoop
from rnm.tools.boundingBox import main as bounding_box_tool
from rnm.tools.visual_io import camera, display_image

dotenv.load_dotenv()


def process_args(args):
    if args.image_path:
        assert Path(args.image_path).exists()
        C.STATIC_IMAGE_PATH = args.image_path
    C.USE_REAL_CAMERA = args.use_real_camera
    C.DEBUG = args.debug
    C.AUDIO_OUTPUT = not args.disable_audio_output
    if args.model:
        C.model = args.model


class Loader:
    file = C.THINKING_WORDS
    with open(file) as f:
        words = f.read().strip().split(", ")

    @classmethod
    def loading(cls):
        def _():
            print(f"{random.choice(cls.words)}...")

        threading.Thread(target=_).start()


def llm(user_input: str):
    """
    Agent creation and invocation function.

    Args:
        user_input (str): The input string from the user. Audio to text input.
    """
    print(f"User input: {user_input}")
    Loader.loading()
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
        model=f"ollama:{C.model}",
        tools=[camera, display_image, bounding_box_tool, *misc_tools.misc_tools],
        # middleware=[TodoListMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        debug=C.DEBUG,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
    )
    print(f">> {result['messages'][-1].content}")


def main():
    process_args(args)
    print("Starting AI Gun - Rick and Morty Edition")
    try:
        inputLoop(llm)
    except KeyboardInterrupt:
        misc_tools.stop()
    # llm("Take a picture using the camera, find the alien faces, and shoot them.")


if __name__ == "__main__":
    main()
