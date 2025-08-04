pub mod core;
pub mod providers;
pub mod unified;

// Re-export core types
pub use core::{Message, ToolCall, Function, ChatStreamItem, PullProgress, ModelInfo, Tool, FallbackToolHandler, AIRequestError, UnifiedModel};

// Main unified interface
pub use unified::UnifiedAI;