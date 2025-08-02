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
    /// Create a client for Ollama
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

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        match &mut self.provider {
            Provider::Ollama(client) => client.add_tool(tool).await,
        }
    }

    pub fn is_fallback_mode(&self) -> bool {
        match &self.provider {
            Provider::Ollama(client) => client.is_fallback_mode(),
        }
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.supports_tool_calls().await,
        }
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request(messages).await,
        }
    }

    pub async fn send_chat_request_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_stream(messages).await,
        }
    }

    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images(messages, image_paths).await,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate(prompt).await,
        }
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate_stream(prompt).await,
        }
    }

    pub async fn list_local_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.list_local_models().await,
        }
    }

    pub async fn show_model_info(&self, model_name: &str) -> Result<ModelInfo, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.show_model_info(model_name).await,
        }
    }

    pub async fn pull_model(&self, model_name: &str) -> Result<(), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model(model_name).await,
        }
    }

    pub async fn pull_model_stream(
        &self,
        model_name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullProgress, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model_stream(model_name).await,
        }
    }

    pub fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        match &self.provider {
            Provider::Ollama(client) => client.handle_tool_calls(tool_calls),
        }
    }

    pub fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        match &self.provider {
            Provider::Ollama(client) => client.process_fallback_response(content),
        }
    }

    // Convenience method to get the model name for display
    pub fn model(&self) -> &str {
        match &self.provider {
            Provider::Ollama(client) => &client.model,
        }
    }

    // Provider-specific methods (for advanced usage)
    pub fn as_ollama(&self) -> Option<&OllamaClient> {
        match &self.provider {
            Provider::Ollama(client) => Some(client),
        }
    }

    pub fn as_ollama_mut(&mut self) -> Option<&mut OllamaClient> {
        match &mut self.provider {
            Provider::Ollama(client) => Some(client),
        }
    }
}