# Actual Workflow

This chart is not rendering properly so the shown on the homepage is a minified version and this one is the detailed version that can be rendered inside a 3rd Party rendering tool. The one used to generate it is [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) in VS Code.

### Image Version
![input 1](../assets/flowchart.png)

### Interactive

```mermaid
flowchart TD
    mic["🎤 always-on mic stream"] --> wake["wake word detection: openWakeWord"]
    wake --> vad["record until silence \n webrtcvad"]
    vad --> stt["speech to text \n faster-whisper"]
    stt -->|transcript| agent{{"LangChain agent \n Ollama"}}

    agent --> cam["camera \n static image or webcam"] --> |image path| agent
    agent --> disp["display image"] --> agent
    agent --> misc["time / date / \ncalculator / stop"] ---> |output| agent

    cam -.->|image path| inner

    subgraph inner["find_aliens_and_shoot"]
        direction TB
        dino["GroundingDINO \n zero-shot face boxes"] --> clip["CLIP \n human vs alien"]
        clip --> merge["Overlapping box detection\nand collapse"]
        merge --> draw["red/green boxes  \n and crosshairs"]
        draw --> out["Save Image"]
    end

    out -.->|saved path| disp
    agent --> inner --> |saved path| agent
    agent ==>|RESET| mic
```
