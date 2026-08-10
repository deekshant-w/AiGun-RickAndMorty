import argparse
import logging

import dotenv
from langchain.agents import create_agent

from rnm.tools import visual_io

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
    SYSTEM_PROMPT = "You are a helpful AI agent that plans before acting and uses tools to interact with the world."
    agent = create_agent(
        model="ollama:ornith:latest",
        tools=[visual_io.camera, visual_io.display_image],
        system_prompt=SYSTEM_PROMPT,
        debug=True,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Take a picture of the world, and display it.",
                }
            ]
        },
    )
    print(result)


if __name__ == "__main__":
    main()
