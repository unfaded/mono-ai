pub mod core;
pub mod providers;
pub mod mono;

// Re-export core types
pub use core::{Message, ToolCall, Function, ChatStreamItem, PullProgress, ModelInfo, Tool, FallbackToolHandler, AIRequestError, MonoModel};

// Main interface
pub use mono::MonoAI;