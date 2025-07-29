# ollama-rust

Rust client for Ollama with streaming tool calls, multimodal support, and full parameter control.

[![Crates.io](https://img.shields.io/crates/v/ollama-rust.svg)](https://crates.io/crates/ollama-rust)
[![Documentation](https://docs.rs/ollama-rust/badge.svg)](https://docs.rs/ollama-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Streaming responses that include tool calls
- Tool functions use doc comments (///) to generate descriptions for the AI
- Full support for all Ollama parameters and endpoints
- Built for production use with proper error handling

## Installation

```bash
cargo add ollama-rust ollama-rust-macros tokio serde_json
```

Note: tokio is included for convenience since all examples use #[tokio::main]. serde_json is needed because the #[tool] macro generates code that references serde_json::Value - without it you'll get "unresolved import" errors.

## Quick Start

```rust
use ollama_rust::{OllamaClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new(
        "http://localhost:11434".to_string(),
        "qwen3:8b".to_string(),
    );

    let messages = vec![Message {
        role: "user".to_string(),
        content: "Hello!".to_string(),
        images: None,
        tool_calls: None,
    }];

    let (response, _) = client.send_chat_request(&messages).await?;
    println!("{}", response);
    Ok(())
}
``` 

## Key Features

### Streaming with Tool Calls

Stream responses while handling function calls in real-time:

```rust
use futures_util::StreamExt;
use ollama_rust::{OllamaClient, Message};
use ollama_rust_macros::tool;

#[tool]
/// Get the current weather for a given location
fn get_weather(location: String) -> String {
    format!("Weather in {}: 72°F and sunny", location)
}

let mut client = OllamaClient::new("http://localhost:11434".to_string(), "qwen3:8b".to_string());
client.add_tool(get_weather_tool());

let mut messages = vec![Message {
    role: "user".to_string(),
    content: "What's the weather in Tokyo?".to_string(),
    images: None,
    tool_calls: None,
}];

let mut stream = client.send_chat_request_stream(&messages).await?;

while let Some(item) = stream.next().await {
    let item = item?;
    
    if !item.content.is_empty() {
        print!("{}", item.content);
    }
    
    if let Some(tool_calls) = item.tool_calls {
        let tool_responses = client.handle_tool_calls(tool_calls);
        // Add tool responses to conversation and continue streaming
        messages.extend(tool_responses);
        
        let mut tool_stream = client.send_chat_request_stream(&messages).await?;
        while let Some(tool_item) = tool_stream.next().await {
            let tool_item = tool_item?;
            if !tool_item.content.is_empty() {
                print!("{}", tool_item.content);
            }
            if tool_item.done { break; }
        }
    }
    
    if item.done { break; }
}
```

### Parameter Control

Fine-tune model behavior with comprehensive options:

```rust
use ollama_rust::OllamaOptions;

let mut options = OllamaOptions::default();
options.temperature = Some(0.8);
options.top_p = Some(0.9);
options.num_predict = Some(500);

let (response, _) = client
    .send_chat_request_with_options(&messages, Some(options))
    .await?;
```

### Multimodal Support

Send images to vision models:

```rust
let messages = vec![Message {
    role: "user".to_string(),
    content: "What's in this image?".to_string(),
    images: None,
    tool_calls: None,
}];

let (response, _) = client
    .send_chat_request_with_images(&messages, vec!["path/to/image.jpg".to_string()])
    .await?;
```

## Example

Run the included example project

```bash
# Interactive chat with tools
cargo run qwen3:8b

# List available models
cargo run list

# Pull a new model
cargo run pull qwen3:8b

# Analyze an image
cargo run image gemma:12b path/to/image.jpg

# Direct text generation
cargo run generate qwen3:8b "Write a haiku about programming"
```

## API Reference

| Method | Purpose |
|--------|---------|
| send_chat_request() | Single chat completion |
| send_chat_request_stream() | Streaming chat with tool calls |
| send_chat_request_with_options() | Chat with custom parameters |
| generate() | Text completion |
| generate_stream() | Streaming text completion |
| list_local_models() | List installed models |
| pull_model() | Download model |

## Available Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| temperature | f32 | Controls creativity. Higher = more creative/random, lower = more focused. Range 0.0-1.0 |
| top_p | f32 | Only consider tokens with cumulative probability up to this value. 0.9 is common |
| top_k | i32 | Only consider the top K most likely tokens. 40-100 typical |
| repeat_penalty | f32 | Penalize repeated words. 1.0 = no penalty, >1.0 reduces repetition |
| num_predict | i32 | Maximum number of tokens to generate in response |
| num_ctx | i32 | Context window size - how much conversation history to remember |
| num_gpu | i32 | Number of GPU layers to use for acceleration |
| seed | i32 | Random seed for reproducible outputs. Same seed = same response |

See [full parameter list](src/lib.rs#L72-L110) for all options. 

All parameters are optional - you don't need to set any to get started.

## Troubleshooting

### Requirements
- [Ollama](https://ollama.ai/) installed
- Have an Ollama model installed

### Streaming Stops Early
Check that your model supports the tool call feature. Some older models may not support function calling.

## License

MIT License