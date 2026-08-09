import argparse
import logging

from langchain.agents import create_agent

from src.rnm.tools import visual_io

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
    logging.info("Starting AI Gun - Rick and Morty Edition")
    SYSTEM_PROMPT = "You are a helpful AI agent that plans before acting and uses tools to interact with the world."
    agent = create_agent(
        model="ollama:qwen3.5:latest",
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
    args = parser.parse_args()
    process_args(args)
    main()
