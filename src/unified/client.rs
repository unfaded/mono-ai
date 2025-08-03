use std::error::Error;
use std::pin::Pin;
use futures_util::Stream;

use crate::core::{Message, ToolCall, ChatStreamItem, PullProgress, ModelInfo, Tool};
use crate::providers::ollama::{OllamaClient, Model};

pub enum Provider {
    Ollama(OllamaClient),
    // Future providers
    // OpenAI(OpenAIClient),
    // Anthropic(AnthropicClient),
}

pub struct UnifiedAI {
    provider: Provider,
}

impl UnifiedAI {
    /// Create Ollama client with endpoint URL and model name
    pub fn ollama(endpoint: String, model: String) -> Self {
        Self {
            provider: Provider::Ollama(OllamaClient::new(endpoint, model)),
        }
    }

    // Future provider constructors
    // pub fn openai(api_key: String, model: String) -> Self {
    //     Self {
    //         provider: Provider::OpenAI(OpenAIClient::new(api_key, model)),
    //     }
    // }
    //
    // pub fn anthropic(api_key: String, model: String) -> Self {
    //     Self {
    //         provider: Provider::Anthropic(AnthropicClient::new(api_key, model)),
    //     }
    // }

    /// Add function tool to client. Automatically enables fallback mode for non-supporting models
    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        match &mut self.provider {
            Provider::Ollama(client) => client.add_tool(tool).await,
        }
    }

    /// Check if client is using fallback tool calling (XML prompting vs native tools)
    pub async fn is_fallback_mode(&self) -> bool {
        match &self.provider {
            Provider::Ollama(client) => client.is_fallback_mode().await,
        }
    }

    /// Enable/disable debug mode to show raw tool call XML in fallback mode
    pub fn set_debug_mode(&mut self, debug: bool) {
        match &mut self.provider {
            Provider::Ollama(client) => client.set_debug_mode(debug),
        }
    }

    /// Check if debug mode is enabled
    pub fn debug_mode(&self) -> bool {
        match &self.provider {
            Provider::Ollama(client) => client.debug_mode(),
        }
    }

    /// Check if model supports native tool calling by examining template
    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.supports_tool_calls().await,
        }
    }

    /// Send chat request with real-time streaming response
    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request(messages).await,
        }
    }

    /// Send chat request without streaming, returns complete response and tool calls
    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_no_stream(messages).await,
        }
    }

    /// Send chat request with images from file paths, returns real-time streaming response
    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images(messages, image_paths).await,
        }
    }

    /// Send chat request with images from file paths, returns complete response and tool calls
    pub async fn send_chat_request_with_images_no_stream(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_no_stream(messages, image_paths).await,
        }
    }

    /// Send chat request with image data from memory, returns real-time streaming response (single image: vec![data], multiple: vec![data1, data2])
    pub async fn send_chat_request_with_image_data(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_data(messages, images_data).await,
        }
    }

    /// Send chat request with image data from memory, returns complete response and tool calls (single image: vec![data], multiple: vec![data1, data2])
    pub async fn send_chat_request_with_image_data_no_stream(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_data_no_stream(messages, images_data).await,
        }
    }

    /// Generate single completion from prompt without conversation context
    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate(prompt).await,
        }
    }

    /// Generate streaming completion from prompt without conversation context
    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate_stream(prompt).await,
        }
    }

    /// List locally installed models (provider-specific operation)
    pub async fn list_local_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.list_local_models().await,
        }
    }

    /// Get detailed model information including template and parameters
    pub async fn show_model_info(&self, model_name: &str) -> Result<ModelInfo, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.show_model_info(model_name).await,
        }
    }

    /// Download model from provider registry (provider-specific operation)
    pub async fn pull_model(&self, model_name: &str) -> Result<(), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model(model_name).await,
        }
    }

    /// Download model with streaming progress updates (provider-specific operation)
    pub async fn pull_model_stream(
        &self,
        model_name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullProgress, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model_stream(model_name).await,
        }
    }

    /// Execute tool calls and return formatted messages for conversation continuation
    pub async fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        match &self.provider {
            Provider::Ollama(client) => client.handle_tool_calls(tool_calls).await,
        }
    }

    /// Parse fallback tool calls from response content and clean XML artifacts
    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        match &self.provider {
            Provider::Ollama(client) => client.process_fallback_response(content).await,
        }
    }

    /// Get current model name for display purposes
    pub fn model(&self) -> &str {
        match &self.provider {
            Provider::Ollama(client) => &client.model,
        }
    }

    /// Access underlying Ollama client for provider-specific operations
    pub fn as_ollama(&self) -> Option<&OllamaClient> {
        match &self.provider {
            Provider::Ollama(client) => Some(client),
        }
    }

    /// Access underlying Ollama client mutably for provider-specific operations
    pub fn as_ollama_mut(&mut self) -> Option<&mut OllamaClient> {
        match &mut self.provider {
            Provider::Ollama(client) => Some(client),
        }
    }
}