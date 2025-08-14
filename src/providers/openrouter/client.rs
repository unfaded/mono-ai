use crate::core::{Message, ChatStreamItem, ToolCall, Tool, MonoModel, TokenUsage, FallbackToolHandler};
use super::types::*;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use futures_util::{StreamExt, Stream};
use std::pin::Pin;
use base64::{Engine as _};

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Content(String),
    ToolCall { id: String, name: String, arguments: String },
    Done,
    Usage(TokenUsage),
}

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub include_usage: bool,
}

pub struct OpenRouterClient {
    client: Client,
    api_key: String,
    pub model: String,
    base_url: String,
    tools: Vec<Tool>,
}

struct OpenRouterStreamProcessor {
    buffer: String,
    accumulating_tool_args: HashMap<usize, String>,
    tool_call_info: HashMap<usize, (String, String)>,
    usage: Option<TokenUsage>,
}

impl OpenRouterStreamProcessor {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            accumulating_tool_args: HashMap::new(),
            tool_call_info: HashMap::new(),
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some(event_end) = self.buffer.find("\n\n") {
            let event_data = self.buffer[..event_end].trim().to_string();
            self.buffer = self.buffer[event_end + 2..].to_string();

            if event_data.starts_with(':') {
                continue;
            }

            if let Some(data) = event_data.strip_prefix("data: ") {
                if data == "[DONE]" {
                    events.push(StreamEvent::Done);
                    break;
                }

                match serde_json::from_str::<OpenRouterResponse>(data) {
                    Ok(response) => {
                        // Extract usage information if available
                        if let Some(usage) = &response.usage {
                            let token_usage = TokenUsage {
                                prompt_tokens: Some(usage.prompt_tokens),
                                completion_tokens: Some(usage.completion_tokens),
                                total_tokens: Some(usage.total_tokens),
                                cost_usd: None, // Will be calculated later in the stream
                            };
                            self.usage = Some(token_usage.clone());
                            events.push(StreamEvent::Usage(token_usage));
                        }
                        
                        if let Some(choice) = response.choices.first() {
                            if let Some(delta) = &choice.delta {
                                
                                // Check content
                                if let Some(content_str) = delta.content.as_str() {
                                    if !content_str.is_empty() {
                                        events.push(StreamEvent::Content(content_str.to_string()));
                                    }
                                }

                                // Check tool calls
                                match &delta.tool_calls {
                                    Some(tool_calls) => {
                                        for (index, tool_call) in tool_calls.iter().enumerate() {
                                            
                                            // Store ID and name when we first see them
                                            if let Some(id) = &tool_call.id {
                                                if let Some(function) = &tool_call.function {
                                                    if let Some(name) = &function.name {
                                                        self.tool_call_info.insert(index, (id.clone(), name.clone()));
                                                    }
                                                }
                                            }
                                            
                                            if let Some(function) = &tool_call.function {
                                                if let Some(args) = &function.arguments {
                                                    let accumulated = self
                                                        .accumulating_tool_args
                                                        .entry(index)
                                                        .or_insert_with(String::new);
                                                    accumulated.push_str(args);

                                                    // Try to parse as JSON
                                                    match serde_json::from_str::<serde_json::Value>(accumulated) {
                                                        Ok(_parsed) => {
                                                            // Use stored ID and name if available
                                                            if let Some((stored_id, stored_name)) = self.tool_call_info.get(&index) {
                                                                events.push(StreamEvent::ToolCall {
                                                                    id: stored_id.clone(),
                                                                    name: stored_name.clone(),
                                                                    arguments: accumulated.clone(),
                                                                });
                                                                self.tool_call_info.remove(&index);
                                                            } else if let Some(id) = &tool_call.id {
                                                                events.push(StreamEvent::ToolCall {
                                                                    id: id.clone(),
                                                                    name: function.name.clone().unwrap_or_default(),
                                                                    arguments: accumulated.clone(),
                                                                });
                                                            }
                                                            self.accumulating_tool_args.remove(&index);
                                                        },
                                                        Err(_) => {
                                                            // JSON parsing failed, continue accumulating
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    None => {
                                        // No tool calls in this chunk
                                    }
                                }
                            }

                            if let Some(finish_reason) = &choice.finish_reason {
                                if !finish_reason.is_empty() {
                                    events.push(StreamEvent::Done);
                                }
                            }
                        }
                    },
                    Err(_) => {
                        // Failed to parse JSON chunk, skip it
                    }
                }
            }
        }

        events
    }
}

impl OpenRouterClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            tools: Vec::new(),
        }
    }

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn std::error::Error>> {
        self.tools.push(tool);
        Ok(())
    }

    pub async fn is_fallback_mode(&self) -> bool {
        if self.tools.is_empty() {
            false // No tools, no fallback needed
        } else {
            // Dynamically check if model supports native tools
            !self.supports_tool_calls().await.unwrap_or(false)
        }
    }

    pub fn set_debug_mode(&mut self, _debug: bool) {
        // OpenRouter debug mode not implemented
    }

    pub fn debug_mode(&self) -> bool {
        false
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn std::error::Error>> {
        // Get model info to check supported parameters
        let response = self
            .client
            .get(&format!("{}/models/{}", self.base_url, self.model))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;
            
        if !response.status().is_success() {
            // If we can't get model info, assume no tool support
            return Ok(false);
        }
        
        let model_info: OpenRouterModel = response.json().await?;
        
        // Check if 'tools' is in the supported_parameters
        if let Some(supported_params) = model_info.supported_parameters {
            Ok(supported_params.contains(&"tools".to_string()))
        } else {
            // If no supported_parameters field, assume no tool support
            Ok(false)
        }
    }

    pub async fn get_usage_for_messages(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        images: &[String],
    ) -> Result<Option<TokenUsage>, Box<dyn std::error::Error>> {
        let openrouter_messages = self.convert_messages(messages, images);
        let openrouter_tools = tools.map(|t| self.convert_tools(t));

        let request = OpenRouterRequest {
            model: self.model.clone(),
            messages: openrouter_messages,
            tools: openrouter_tools,
            tool_choice: None,
            stream: Some(false), // Non-streaming to get usage
            max_tokens: Some(1), // Minimal tokens since we just want usage
            temperature: Some(0.7),
            stream_options: None, // Not needed for non-streaming
        };

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            // Don't fail the main request if usage call fails
            return Ok(None);
        }

        let openrouter_response: OpenRouterResponse = response.json().await?;
        
        if let Some(usage) = openrouter_response.usage {
            Ok(Some(TokenUsage {
                prompt_tokens: Some(usage.prompt_tokens),
                completion_tokens: Some(usage.completion_tokens),
                total_tokens: Some(usage.total_tokens),
                cost_usd: None,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_available_models(&self) -> Result<Vec<MonoModel>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get(&format!("{}/models", self.base_url))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("OpenRouter API error: {}", error_text).into());
        }

        let openrouter_response: OpenRouterModelsResponse = response.json().await?;
        
        let models = openrouter_response.data
            .into_iter()
            .map(|model| MonoModel {
                id: model.id,
                name: model.name,
                provider: "OpenRouter".to_string(),
                size: None,
                created: None,
            })
            .collect();

        Ok(models)
    }

    fn convert_messages(&self, messages: &[Message], images: &[String]) -> Vec<OpenRouterMessage> {
        let mut openrouter_messages = Vec::new();
        let mut last_tool_call_info: Option<(String, String)> = None;

        for message in messages {
            // Track tool call IDs and names from assistant messages
            if message.role == "assistant" && message.tool_calls.is_some() {
                if let Some(tool_calls) = &message.tool_calls {
                    if let Some(first_call) = tool_calls.first() {
                        if let Some(id) = &first_call.id {
                            let name = first_call.function.name.clone();
                            last_tool_call_info = Some((id.clone(), name.clone()));
                        }
                    }
                }
            }

            // Handle tool result messages using OpenRouter's standard format
            if message.role == "tool" {
                if let Some((tool_use_id, tool_name)) = &last_tool_call_info {                    
                    let msg = OpenRouterMessage {
                        role: "tool".to_string(),
                        content: serde_json::Value::String(message.content.clone()),
                        name: Some(tool_name.clone()),
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id.clone()),
                    };
                    openrouter_messages.push(msg);
                    continue;
                }
            }

            let mut content_items = Vec::new();

            if !message.content.is_empty() {
                content_items.push(json!({
                    "type": "text",
                    "text": message.content
                }));
            }

            if message.role == "user" && !images.is_empty() {
                for image in images {
                    content_items.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{}", image)
                        }
                    }));
                }
            }

            let content = if content_items.len() == 1 && content_items[0]["type"] == "text" {
                // Use simple string for basic text messages
                serde_json::Value::String(message.content.clone())
            } else if content_items.is_empty() {
                // Empty content
                serde_json::Value::String(message.content.clone())
            } else {
                // Complex content with images
                json!(content_items)
            };

            let tool_calls = if let Some(calls) = &message.tool_calls {
                Some(calls.iter().map(|call| OpenRouterToolCall {
                    id: call.id.clone(),
                    call_type: Some("function".to_string()),
                    function: Some(OpenRouterFunctionCall {
                        name: Some(call.function.name.clone()),
                        arguments: Some(serde_json::to_string(&call.function.arguments).unwrap_or_default()),
                    }),
                }).collect())
            } else {
                None
            };

            openrouter_messages.push(OpenRouterMessage {
                role: message.role.clone(),
                content,
                name: None,
                tool_calls,
                tool_call_id: None,
            });
        }

        openrouter_messages
    }

    fn convert_tools(&self, tools: &[Tool]) -> Vec<OpenRouterTool> {
        tools
            .iter()
            .map(|tool| OpenRouterTool {
                tool_type: "function".to_string(),
                function: OpenRouterFunction {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    parameters: tool.parameters.clone(),
                },
            })
            .collect()
    }

    pub async fn chat_completion(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        images: Vec<String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let openrouter_messages = self.convert_messages(&messages, &images);
        let openrouter_tools = tools.as_ref().map(|t| self.convert_tools(t));

        let request = OpenRouterRequest {
            model: self.model.clone(),
            messages: openrouter_messages,
            tools: openrouter_tools,
            tool_choice: None,
            stream: Some(false),
            max_tokens: Some(4096),
            temperature: Some(0.7),
            stream_options: None, // Not needed for non-streaming
        };

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("OpenRouter API error: {}", error_text).into());
        }

        let openrouter_response: OpenRouterResponse = response.json().await?;

        if let Some(choice) = openrouter_response.choices.first() {
            if let Some(message) = &choice.message {
                if let Some(content) = message.content.as_str() {
                    return Ok(content.to_string());
                }
            }
        }

        Err("No content in response".into())
    }

    pub async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        _options: StreamOptions,
        images: Vec<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, String>> + Send>>, Box<dyn std::error::Error>> {
        let openrouter_messages = self.convert_messages(&messages, &images);
        let openrouter_tools = tools.as_ref().map(|t| self.convert_tools(t));

        let request = OpenRouterRequest {
            model: self.model.clone(),
            messages: openrouter_messages,
            tools: openrouter_tools,
            tool_choice: None,
            stream: Some(true),
            max_tokens: Some(4096),
            temperature: Some(0.7),
            stream_options: Some(super::types::OpenRouterStreamOptions { include_usage: true }),
        };

        let response = self
            .client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("OpenRouter API error: {}", error_text).into());
        }

        let stream = response.bytes_stream();
        let mut processor = OpenRouterStreamProcessor::new();

        let event_stream = stream.map(move |chunk| {
            match chunk {
                Ok(bytes) => {
                    let chunk_str = String::from_utf8_lossy(&bytes);
                    let events = processor.process_chunk(&chunk_str);
                    events
                }
                Err(e) => {
                    vec![StreamEvent::Content(format!("Network error: {}", e))]
                }
            }
        })
        .map(|events| futures_util::stream::iter(events.into_iter().map(Ok)))
        .flatten();

        Ok(Box::pin(event_stream))
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn futures_util::Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn std::error::Error>> {
        let tools = if !self.tools.is_empty() {
            Some(self.tools.iter().map(|tool| Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                function: Box::new(|_| "Not implemented".to_string()),
            }).collect())
        } else {
            None
        };

        let images: Vec<String> = messages
            .iter()
            .filter_map(|m| m.images.as_ref())
            .flatten()
            .cloned()
            .collect();

        let stream_options = StreamOptions { include_usage: true };
        let event_stream = self.chat_completion_stream(messages.to_vec(), tools, stream_options, images).await?;

        // Store client info for usage request
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let messages_for_usage = messages.to_vec();
        
        let mapped_stream = event_stream.then(move |event| {
            let api_key = api_key.clone();
            let model = model.clone();
            let base_url = base_url.clone();
            let client = client.clone();
            let messages_for_usage = messages_for_usage.clone();
            
            async move {
                match event {
                    Ok(StreamEvent::Content(content)) => Ok(ChatStreamItem {
                        content,
                        tool_calls: None,
                        done: false,
                        usage: None,
                    }),
                    Ok(StreamEvent::ToolCall { id, name, arguments }) => {
                        Ok(ChatStreamItem {
                            content: String::new(),
                            tool_calls: Some(vec![ToolCall {
                                id: Some(id),
                                function: crate::core::Function { name, arguments: serde_json::from_str(&arguments).unwrap_or(serde_json::Value::Null) },
                            }]),
                            done: false,
                            usage: None,
                        })
                    }
                    Ok(StreamEvent::Usage(usage)) => Ok(ChatStreamItem {
                        content: String::new(),
                        tool_calls: None,
                        done: false,
                        usage: Some(usage),
                    }),
                    Ok(StreamEvent::Done) => {
                        // Make a quick usage request when stream is done
                        let usage = get_usage_estimate(&client, &api_key, &base_url, &model, &messages_for_usage).await;
                        
                        Ok(ChatStreamItem {
                            content: String::new(),
                            tool_calls: None,
                            done: true,
                            usage,
                        })
                    },
                    Err(e) => Err(e),
                }
            }
        });

        Ok(Box::pin(mapped_stream))
    }

    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn std::error::Error>> {
        let tools = if !self.tools.is_empty() {
            Some(self.tools.iter().map(|tool| Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                function: Box::new(|_| "Not implemented".to_string()),
            }).collect())
        } else {
            None
        };

        let images: Vec<String> = messages
            .iter()
            .filter_map(|m| m.images.as_ref())
            .flatten()
            .cloned()
            .collect();

        let response = self.chat_completion(messages.to_vec(), tools, images).await?;
        
        Ok((response, None))
    }

    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<Pin<Box<dyn futures_util::Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn std::error::Error>> {
        // For OpenRouter, encode images and add them to the messages
        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            let mut encoded_images = Vec::new();
            for image_path in image_paths {
                let image_data = tokio::fs::read(&image_path).await
                    .map_err(|e| format!("Failed to read image file {}: {}", image_path, e))?;
                let encoded = base64::engine::general_purpose::STANDARD.encode(&image_data);
                encoded_images.push(encoded);
            }
            last_message.images = Some(encoded_images);
        }
        self.send_chat_request(&messages_with_images).await
    }

    pub async fn send_chat_request_with_images_no_stream(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn std::error::Error>> {
        // For OpenRouter, encode images and add them to the messages
        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            let mut encoded_images = Vec::new();
            for image_path in image_paths {
                let image_data = tokio::fs::read(&image_path).await
                    .map_err(|e| format!("Failed to read image file {}: {}", image_path, e))?;
                let encoded = base64::engine::general_purpose::STANDARD.encode(&image_data);
                encoded_images.push(encoded);
            }
            last_message.images = Some(encoded_images);
        }
        self.send_chat_request_no_stream(&messages_with_images).await
    }

    pub async fn send_chat_request_with_images_data(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<Pin<Box<dyn futures_util::Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn std::error::Error>> {
        // For OpenRouter, encode images and add them to the messages
        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            let mut encoded_images = Vec::new();
            for image_data in images_data {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&image_data);
                encoded_images.push(encoded);
            }
            last_message.images = Some(encoded_images);
        }
        self.send_chat_request(&messages_with_images).await
    }

    pub async fn send_chat_request_with_images_data_no_stream(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn std::error::Error>> {
        // For OpenRouter, encode images and add them to the messages
        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            let mut encoded_images = Vec::new();
            for image_data in images_data {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&image_data);
                encoded_images.push(encoded);
            }
            last_message.images = Some(encoded_images);
        }
        self.send_chat_request_no_stream(&messages_with_images).await
    }

    pub async fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        // Similar to other providers, execute tool calls and return formatted messages
        let mut messages = Vec::new();
        for tool_call in tool_calls {
            let result = self.execute_tool_call(&tool_call).await;
            messages.push(Message {
                role: "tool".to_string(),
                content: result,
                images: None,
                tool_calls: None,
            });
        }
        messages
    }

    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        let is_fallback = self.is_fallback_mode().await;
        if !is_fallback {
            return (content.to_string(), None);
        }

        FallbackToolHandler::process_fallback_response(content)
    }

    async fn execute_tool_call(&self, tool_call: &ToolCall) -> String {
        // Find the tool in our tools list
        if let Some(tool) = self.tools.iter().find(|t| t.name == tool_call.function.name) {
            // Execute the tool function
            return (tool.function)(tool_call.function.arguments.clone());
        }
        format!("Tool {} not found or invalid arguments", tool_call.function.name)
    }
}

// Helper function to get model pricing from OpenRouter API
async fn get_model_pricing(
    client: &Client,
    model: &str,
) -> Option<(f64, f64)> {
    let response = client
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await;

    if let Ok(response) = response {
        if response.status().is_success() {
            if let Ok(models_response) = response.json::<serde_json::Value>().await {
                if let Some(data) = models_response["data"].as_array() {
                    for model_data in data {
                        if let Some(id) = model_data["id"].as_str() {
                            if id == model {
                                if let Some(pricing) = model_data["pricing"].as_object() {
                                    let prompt_price = pricing["prompt"].as_str()
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .unwrap_or(0.0);
                                    let completion_price = pricing["completion"].as_str()
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .unwrap_or(0.0);
                                    return Some((prompt_price, completion_price));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    None
}

// Helper function to get usage information
async fn get_usage_estimate(
    client: &Client,
    api_key: &str,
    base_url: &str,
    model: &str,
    messages: &[Message],
) -> Option<TokenUsage> {
    // Convert messages to OpenRouter format
    let openrouter_messages: Vec<super::types::OpenRouterMessage> = messages
        .iter()
        .map(|msg| super::types::OpenRouterMessage {
            role: msg.role.clone(),
            content: serde_json::Value::String(msg.content.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        })
        .collect();

    let request = super::types::OpenRouterRequest {
        model: model.to_string(),
        messages: openrouter_messages,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        max_tokens: Some(1), // Minimal tokens since we just want usage
        temperature: Some(0.7),
        stream_options: None,
    };

    let response = client
        .post(&format!("{}/chat/completions", base_url))
        .header("Authorization", &format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await;

    if let Ok(response) = response {
        if response.status().is_success() {
            if let Ok(openrouter_response) = response.json::<super::types::OpenRouterResponse>().await {
                if let Some(usage) = openrouter_response.usage {
                    // Get pricing information for cost calculation
                    let cost_usd = if let Some((prompt_price, completion_price)) = get_model_pricing(client, model).await {
                        let prompt_cost = usage.prompt_tokens as f64 * prompt_price;
                        let completion_cost = usage.completion_tokens as f64 * completion_price;
                        Some(prompt_cost + completion_cost)
                    } else {
                        None
                    };

                    return Some(TokenUsage {
                        prompt_tokens: Some(usage.prompt_tokens),
                        completion_tokens: Some(usage.completion_tokens),
                        total_tokens: Some(usage.total_tokens),
                        cost_usd,
                    });
                }
            }
        }
    }
    
    None
}