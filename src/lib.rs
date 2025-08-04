pub mod core;
pub mod providers;
pub mod unified;

// Re-export core types
pub use core::{Message, ToolCall, Function, ChatStreamItem, PullProgress, ModelInfo, Tool, FallbackToolHandler, AIRequestError, UnifiedModel};

// Re-export providers for direct access
pub use providers::ollama::{OllamaClient, OllamaOptions, Model};

// Main unified interface
pub use unified::UnifiedAI;