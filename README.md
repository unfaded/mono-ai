# Unified AI

A provider-agnostic Rust library for interacting with AI services. Switch between Ollama, Anthropic, OpenAI, and OpenRouter with identical code.

## Features

- Universal Interface: Same API across all AI providers
- Streaming Support: Real-time response streaming
- Vision Capabilities: Image analysis and multimodal conversations  
- Tool Calling: Function execution with automatic fallback for unsupported models
- Tool Macros: Automatically converts function doc comments into AI tool descriptions
- Model Management: List, pull, and inspect available models
- Async/Await: Full async support with proper error handling

## Supported Providers

Ollama, Anthropic, OpenAI, and OpenRouter all support chat, streaming, vision, tools, and model management through the same unified interface.

## Quick Start

Add library:
```bash
cargo add unified-ai unified-ai-macros
```

Add dependencies:

```bash
cargo add tokio --features full 
cargo add futures-util serde_json
```

See the `examples/` directory for complete working examples that demonstrate the library's capabilities.

## Environment Variables

Set API keys via environment variables for the providers you want to use

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Examples

The `examples/` directory contains three comprehensive examples demonstrating all library features, and outside of the constructor, all the code stays the same no matter the model

### Chat

Interactive chat application with provider selection menu (Ollama, Anthropic, OpenAI, OpenRouter) and automatic model discovery. Implements streaming chat responses, tool calling with custom functions (weather lookup, password generation), conversation history management, and error handling.

### Vision Chat

Multimodal chat application for image analysis. Takes image file path as command line argument, performs initial analysis, then enables interactive conversation about the image. Handles base64 encoding, message formatting, conversation context preservation, streaming responses, and tool calls across all vision-capable models & providers.

### Ollama Management

Model management utility for Ollama instances. pulling models from registry with progress tracking, model inspection (templates, parameters), and lifecycle management.

Run examples:

```bash
cd examples/chat && cargo run
cd examples/chat-vision && cargo run path/to/image.jpg
cd examples/ollama-management && cargo run
```

## API Reference

### Creating Clients

Besides constructing the client, the rest of the code is provider agnostic.
```rust
// Local Ollama instance
let client = UnifiedAI::ollama("http://localhost:11434".to_string(), "qwen3:8b".to_string());

// Cloud providers
let client = UnifiedAI::openai(api_key, "gpt-4".to_string());
let client = UnifiedAI::anthropic(api_key, "claude-3-sonnet-20240229".to_string());
let client = UnifiedAI::openrouter(api_key, "anthropic/claude-sonnet-4".to_string());
```

### Core Methods

#### Chat Methods
- `send_chat_request(&messages)` - Streaming chat
- `send_chat_request_no_stream(&messages)` - Complete response
- `generate(prompt)` - Simple completion
- `generate_stream(prompt)` - Streaming completion

#### Vision Methods  
- `send_chat_request_with_images(&messages, image_paths)` - Chat with images from files
- `send_chat_request_with_image_data(&messages, image_data)` - Chat with image bytes
- `encode_image_file(path)` - Encode image file to base64
- `encode_image_data(bytes)` - Encode image bytes to base64

#### Tool Methods
- `add_tool(tool)` - Add function tool
- `handle_tool_calls(tool_calls)` - Execute tools and format responses
- `supports_tool_calls()` - Check native tool support
- `is_fallback_mode()` - Check if using XML fallback
- `process_fallback_response(content)` - Parse fallback tool calls

#### Model Methods
- `get_available_models()` - List available models (unified format)

#### Ollama Management
- `show_model_info(model)` - Get model details (Ollama only)  
- `pull_model(model)` - Download model (Ollama only)
- `pull_model_stream(model)` - Download with progress (Ollama only)

### Tool Definition

Use the `#[tool]` macro to define tool functions

```rust
use unified_ai_macros::tool;

/// The AI will see this doc comment
/// Describe what your tool does and its purpose here
/// The macro automatically provides parameter names, types, and marks all as required
/// You should explain what the function returns and provide usage guidance
#[tool]
fn my_function(param1: String, param2: i32) -> String {
    format!("Got {} and {}", param1, param2)
}

// Add to client
client.add_tool(my_function_tool()).await?;
```

## Advanced Features

### Fallback Tool Calling

Models without native tool support automatically use XML-based fallbacks, if you want to know if it's using it or not, feel free to use the is_fallback_mode function

```rust
if client.is_fallback_mode().await {
    println!("Using XML fallback for tools");
}

// Enable debug mode to see raw XML
client.set_debug_mode(true);
```

## License

MIT License

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.