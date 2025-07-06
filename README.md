## A Comprehensive Ollama Integration in Rust
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Example](#example)

## Overview
- Streaming responses, with tool calls
- Optionally add an image for vision models
- Pull models
- List local models
- Show model information (License, Modelfile, Parameters, Template)

## Requirements
- Ensure you have ollama installed, for installation instructions follow: https://github.com/ollama/ollama?tab=readme-ov-file#ollama

## Usage
To get started and add it to your project

```
cargo add ollama-rust ollama-rust-macros
```

Also make sure to add these dependencies, to avoid tool macro errors
```
cargo add serde_json tokio
```

If you would like a more complete example to experiment with, feel free to clone the repo and try out the example that I included, then go over to [Example](#Example)

I will add more detailed Usage instructions soon.

## Example
After you're inside the example directory (ollama-rust/example) you can start chatting with it by simply running, you can also ask it to use tools which I've included 2, a simple generate_password tool and a get_weather example tool that should always return that it's sunny to the AI

```
cargo run <model_name>
```

List the models you have installed by running
```
cargo run list
```

Pull a model (e.g. if the user doesn't have it yet)
```
cargo run pull <model_name>
```

Send an image (if your model is a vision model), the model will tell you what's in the image
```
cargo run image <model_name> <image_directory>
```
