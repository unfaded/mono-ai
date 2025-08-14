use futures_util::{Stream, StreamExt};
use reqwest::Client;
use std::error::Error;
use std::pin::Pin;
use std::collections::HashMap;
use bytes::Bytes;

use crate::core::{Message, ToolCall, ChatStreamItem, Tool, TokenUsage};
use super::types::*;

// Manual OpenAI model pricing function (based on official OpenAI pricing)
fn get_openai_model_pricing(model: &str) -> (f64, f64) {
    match model {
        // GPT-5 series
        "gpt-5" | "gpt-5-2025-08-07" | "gpt-5-chat-latest" => (1.250e-6, 10.000e-6), // $1.250/1M input, $10.000/1M output
        "gpt-5-mini" | "gpt-5-mini-2025-08-07" => (0.250e-6, 2.000e-6), // $0.250/1M input, $2.000/1M output
        "gpt-5-nano" | "gpt-5-nano-2025-08-07" => (0.050e-6, 0.400e-6), // $0.050/1M input, $0.400/1M output
        
        // GPT-4.1 series
        "gpt-4.1" | "gpt-4.1-2025-04-14" => (3.00e-6, 12.00e-6), // $3.00/1M input, $12.00/1M output
        "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14" => (0.80e-6, 3.20e-6), // $0.80/1M input, $3.20/1M output
        "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14" => (0.20e-6, 0.80e-6), // $0.20/1M input, $0.80/1M output
        
        // Realtime API
        "gpt-4o" | "gpt-4o-2024-05-13" | "gpt-4o-2024-08-06" | "gpt-4o-2024-11-20" => (5.00e-6, 20.00e-6), // $5.00/1M input, $20.00/1M output
        "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => (0.60e-6, 2.40e-6), // $0.60/1M input, $2.40/1M output
        
        // Image Generation API
        "gpt-image-1" => (5.00e-6, 0.0), // $5.00/1M input, no output tokens
        
        // o4-mini reinforcement fine-tuning
        "o4-mini" => (4.00e-6, 16.00e-6), // $4.00/1M input, $16.00/1M output
        
        // Default fallback for unknown models
        _ => (0.0, 0.0),
    }
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    pub model: String,
    tools: Vec<Tool>,
}

impl OpenAIClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            tools: Vec::new(),
        }
    }

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        self.tools.push(tool);
        Ok(())
    }

    pub async fn is_fallback_mode(&self) -> bool {
        false // OpenAI has native tool support
    }

    pub fn set_debug_mode(&mut self, _debug: bool) {
        // OpenAI debug mode not yet implemented
    }

    pub fn debug_mode(&self) -> bool {
        false
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        Ok(true) // OpenAI models support native tool calling
    }

    pub async fn get_available_models(&self) -> Result<Vec<OpenAIModel>, Box<dyn Error>> {
        let response = self
            .client
            .get("https://api.openai.com/v1/models")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("OpenAI API error: {}", error_text).into());
        }

        let models_response: OpenAIModelsResponse = response.json().await?;
        Ok(models_response.data)
    }

    fn convert_to_openai_message(&self, message: &Message) -> OpenAIMessage {
        // Check if this is a tool result message
        if message.role == "tool" {
            // For OpenAI, tool results need tool_call_id and content
            // We'll extract the tool_call_id from our encoded format if present
            let (tool_call_id, content) = if message.content.starts_with("TOOL_RESULT:") {
                let parts: Vec<&str> = message.content.splitn(3, ':').collect();
                if parts.len() == 3 {
                    (Some(parts[1].to_string()), parts[2].to_string())
                } else {
                    (None, message.content.clone())
                }
            } else {
                (None, message.content.clone())
            };

            return OpenAIMessage {
                role: Some(message.role.clone()),
                content: Some(serde_json::Value::String(content)),
                tool_calls: None,
                tool_call_id,
            };
        }

        // Convert tool calls if present
        let tool_calls = if let Some(tc) = &message.tool_calls {
            Some(tc.iter().map(|call| {
                OpenAIToolCall {
                    id: Some(call.id.clone().unwrap_or_else(|| format!("call_{}", "generated_id"))),
                    call_type: Some("function".to_string()),
                    function: OpenAIFunction {
                        name: Some(call.function.name.clone()),
                        arguments: Some(serde_json::to_string(&call.function.arguments).unwrap_or_default()),
                    },
                }
            }).collect())
        } else {
            None
        };

        // Handle vision messages with images for OpenAI's structured content format
        let content = if let Some(ref images) = message.images {
            if !images.is_empty() {
                // Create structured content array for OpenAI vision API
                let mut content_items = vec![];
                
                // Add text content
                if !message.content.is_empty() {
                    content_items.push(serde_json::json!({
                        "type": "text",
                        "text": message.content
                    }));
                }
                
                // Add image content in OpenAI's base64 format
                for image in images {
                    content_items.push(serde_json::json!({
                        "type": "image_url", 
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{}", image)
                        }
                    }));
                }
                
                Some(serde_json::Value::Array(content_items))
            } else {
                Some(serde_json::Value::String(message.content.clone()))
            }
        } else {
            Some(serde_json::Value::String(message.content.clone()))
        };

        OpenAIMessage {
            role: Some(message.role.clone()),
            content,
            tool_calls,
            tool_call_id: None,
        }
    }

    fn convert_tools_to_openai(&self) -> Vec<OpenAITool> {
        self.tools
            .iter()
            .map(|tool| {
                // Ensure the parameters have additionalProperties: false for OpenAI compatibility
                let mut parameters = tool.parameters.clone();
                if let Some(obj) = parameters.as_object_mut() {
                    obj.insert("additionalProperties".to_string(), serde_json::Value::Bool(false));
                }
                
                OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIToolFunction {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters,
                    },
                }
            })
            .collect()
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        let openai_messages: Vec<OpenAIMessage> = messages
            .iter()
            .map(|msg| self.convert_to_openai_message(msg))
            .collect();

        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: openai_messages,
            temperature: None,
            // Use max_completion_tokens for o1 models, max_tokens for others
            max_tokens: if self.model.contains("o1") { None } else { Some(4096) },
            max_completion_tokens: if self.model.contains("o1") { Some(4096) } else { None },
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.convert_tools_to_openai())
            },
            stream: Some(true),
            stream_options: Some(OpenAIStreamOptions { include_usage: true }),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("OpenAI API error: {}", error_text).into());
        }

        let stream = response.bytes_stream();
        
        // Create a stateful stream processor with model for pricing
        Ok(Box::pin(OpenAIStreamProcessor::new(Box::pin(stream), self.model.clone())))
    }

    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut stream = self.send_chat_request(messages).await?;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            if !item.content.is_empty() {
                full_response.push_str(&item.content);
            }
            if let Some(tc) = item.tool_calls {
                tool_calls = Some(tc);
            }
            if item.done {
                return Ok((full_response, tool_calls));
            }
        }
        Ok((full_response, tool_calls))
    }

    pub async fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        let mut tool_responses = Vec::new();
        for tool_call in tool_calls {
            if let Some(tool) = self
                .tools
                .iter()
                .find(|t| t.name == tool_call.function.name)
            {
                let result = (tool.function)(tool_call.function.arguments.clone());
                
                // Use the tool call ID if available, otherwise use "unknown"
                let tool_id = tool_call.id.unwrap_or_else(|| "unknown".to_string());
                
                // Create a message that can be identified as a tool result
                // Use the encoded format: TOOL_RESULT:tool_id:result_content
                tool_responses.push(Message {
                    role: "tool".to_string(),
                    content: format!("TOOL_RESULT:{}:{}", tool_id, result),
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }

    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        // OpenAI doesn't need fallback processing since it has native tool support
        (content.to_string(), None)
    }

    // Get OpenAI model pricing (input cost per token, output cost per token)
    fn get_model_pricing(&self) -> (f64, f64) {
        get_openai_model_pricing(&self.model)
    }

    // Calculate cost based on token usage
    fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let (input_price, output_price) = get_openai_model_pricing(&self.model);
        (prompt_tokens as f64 * input_price) + (completion_tokens as f64 * output_price)
    }
}

// Custom stream processor for OpenAI streaming responses
struct OpenAIStreamProcessor {
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    accumulated_content: String,
    accumulated_tool_calls: HashMap<usize, ToolCall>,
    // Track tool arguments being accumulated: tool_index -> accumulated_json_string
    accumulating_tool_args: HashMap<usize, String>,
    // Buffer for incomplete SSE events that span chunk boundaries
    buffer: String,
    done: bool,
    usage: Option<TokenUsage>,
    model: String,
}

impl OpenAIStreamProcessor {
    fn new(stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>, model: String) -> Self {
        Self {
            stream,
            accumulated_content: String::new(),
            accumulated_tool_calls: HashMap::new(),
            accumulating_tool_args: HashMap::new(),
            buffer: String::new(),
            done: false,
            usage: None,
            model,
        }
    }

    // Get OpenAI model pricing (input cost per token, output cost per token)
    fn get_model_pricing(&self) -> (f64, f64) {
        get_openai_model_pricing(&self.model)
    }

    // Calculate cost based on token usage
    fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let (input_price, output_price) = get_openai_model_pricing(&self.model);
        (prompt_tokens as f64 * input_price) + (completion_tokens as f64 * output_price)
    }
}

impl Stream for OpenAIStreamProcessor {
    type Item = Result<ChatStreamItem, String>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.done {
            return std::task::Poll::Ready(None);
        }

        loop {
            match self.stream.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(chunk_result)) => {
                    match chunk_result {
                        Ok(chunk) => {
                            let chunk_str = String::from_utf8_lossy(&chunk);
                            
                            // Add new chunk to buffer
                            self.buffer.push_str(&chunk_str);
                            
                            // Process complete SSE events from buffer
                            while let Some(event_end) = self.buffer.find("\n\n") {
                                let event = self.buffer[..event_end].to_string();
                                self.buffer = self.buffer[event_end + 2..].to_string(); // Remove processed event + \n\n
                                
                                // Parse each line in the event
                                for line in event.lines() {
                                    if line.starts_with("data: ") {
                                        let json_str = &line[6..]; // Remove "data: " prefix
                                    
                                    if json_str == "[DONE]" {
                                        self.done = true;
                                        let final_tool_calls = if !self.accumulated_tool_calls.is_empty() {
                                            let mut tool_calls = Vec::new();
                                            for (i, mut tool_call) in self.accumulated_tool_calls.clone() {
                                                // Parse the accumulated argument string
                                                if let Some(args_str) = self.accumulating_tool_args.get(&i) {
                                                    if !args_str.is_empty() {
                                                        if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                                            tool_call.function.arguments = args;
                                                        }
                                                    }
                                                }
                                                tool_calls.push(tool_call);
                                            }
                                            Some(tool_calls)
                                        } else {
                                            None
                                        };
                                        
                                        return std::task::Poll::Ready(Some(Ok(ChatStreamItem {
                                            content: String::new(),
                                            tool_calls: final_tool_calls,
                                            done: true,
                                            usage: self.usage.clone(),
                                        })));
                                    }
                                    
                                    match serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                                        Ok(chunk) => {
                                            // Extract usage information if available
                                            if let Some(usage) = &chunk.usage {
                                                let cost_usd = Some(self.calculate_cost(usage.prompt_tokens, usage.completion_tokens));
                                                self.usage = Some(TokenUsage {
                                                    prompt_tokens: Some(usage.prompt_tokens),
                                                    completion_tokens: Some(usage.completion_tokens),
                                                    total_tokens: Some(usage.total_tokens),
                                                    cost_usd,
                                                });
                                            }
                                            
                                            if let Some(choice) = chunk.choices.first() {
                                                if let Some(delta) = &choice.delta {
                                                    let mut content = String::new();
                                                    let mut has_tool_calls = false;
                                                    
                                                    // Handle content delta
                                                    if let Some(delta_content) = &delta.content {
                                                        if let Some(text) = delta_content.as_str() {
                                                            content = text.to_string();
                                                            self.accumulated_content.push_str(&content);
                                                        }
                                                    }
                                                    
                                                    // Handle tool call deltas
                                                    if let Some(tool_calls) = &delta.tool_calls {
                                                        has_tool_calls = true;
                                                        for (i, tool_call) in tool_calls.iter().enumerate() {
                                                            // Ensure tool call entry exists
                                                            if !self.accumulated_tool_calls.contains_key(&i) {
                                                                self.accumulated_tool_calls.insert(i, ToolCall {
                                                                    id: tool_call.id.clone(),
                                                                    function: crate::core::Function {
                                                                        name: tool_call.function.name.clone().unwrap_or_default(),
                                                                        arguments: serde_json::Value::Null,
                                                                    },
                                                                });
                                                            }
                                                            
                                                            // Accumulate function arguments as string chunks
                                                            if let Some(ref args_str) = tool_call.function.arguments {
                                                                if !args_str.is_empty() {
                                                                    let accumulated_args = self.accumulating_tool_args.entry(i).or_insert_with(String::new);
                                                                    accumulated_args.push_str(args_str);
                                                                }
                                                            }
                                                            
                                                            // Update name if provided
                                                            if let Some(ref name) = tool_call.function.name {
                                                                if !name.is_empty() {
                                                                    if let Some(entry) = self.accumulated_tool_calls.get_mut(&i) {
                                                                        entry.function.name = name.clone();
                                                                    }
                                                                }
                                                            }
                                                            
                                                            // Update ID if provided
                                                            if let Some(ref id) = tool_call.id {
                                                                if !id.is_empty() {
                                                                    if let Some(entry) = self.accumulated_tool_calls.get_mut(&i) {
                                                                        entry.id = Some(id.clone());
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    
                                                    if !content.is_empty() || has_tool_calls {
                                                        return std::task::Poll::Ready(Some(Ok(ChatStreamItem {
                                                            content,
                                                            tool_calls: None, // Don't return partial tool calls
                                                            done: false,
                                                            usage: None,
                                                        })));
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            return std::task::Poll::Ready(Some(Err(format!("JSON parse error: {}", e))));
                                        }
                                    }
                                    } // End of line processing
                                } // End of event.lines() loop
                            } // End of while let Some(event_end) loop
                        }
                        Err(e) => {
                            return std::task::Poll::Ready(Some(Err(format!("Stream error: {}", e))));
                        }
                    }
                }
                std::task::Poll::Ready(None) => {                    
                    // Process any remaining data in the buffer before ending
                    if !self.buffer.is_empty() {
                        let buffer_clone = self.buffer.clone();
                        for line in buffer_clone.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..];
                                
                                if json_str == "[DONE]" {
                                    // Stream done signal found in buffer
                                } else if !json_str.is_empty() {
                                    // Process this final chunk
                                    match serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                                        Ok(chunk) => {
                                            if let Some(choice) = chunk.choices.first() {
                                                if let Some(delta) = &choice.delta {
                                                    if let Some(tool_calls) = &delta.tool_calls {
                                                        for (i, tool_call) in tool_calls.iter().enumerate() {
                                                            if let Some(ref args_str) = tool_call.function.arguments {
                                                                if !args_str.is_empty() {
                                                                    let accumulated_args = self.accumulating_tool_args.entry(i).or_insert_with(String::new);
                                                                    accumulated_args.push_str(args_str);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Err(_) => {
                                            // Failed to parse final buffer JSON, ignore
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Process any remaining data in buffer before ending
                    let buffer_content = self.buffer.clone();
                    if !buffer_content.is_empty() {
                        for line in buffer_content.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..];
                                if json_str != "[DONE]" && !json_str.is_empty() {
                                    if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                                        if let Some(usage) = &chunk.usage {
                                            let cost_usd = Some(self.calculate_cost(usage.prompt_tokens, usage.completion_tokens));
                                            self.usage = Some(TokenUsage {
                                                prompt_tokens: Some(usage.prompt_tokens),
                                                completion_tokens: Some(usage.completion_tokens),
                                                total_tokens: Some(usage.total_tokens),
                                                cost_usd,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    self.done = true;
                    let final_tool_calls = if !self.accumulated_tool_calls.is_empty() {
                        let mut tool_calls = Vec::new();
                        for (i, mut tool_call) in self.accumulated_tool_calls.clone() {
                            // Parse the accumulated argument string when stream ends
                            if let Some(args_str) = self.accumulating_tool_args.get(&i) {
                                if !args_str.is_empty() {
                                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                                        tool_call.function.arguments = args;
                                    }
                                }
                            }
                            tool_calls.push(tool_call);
                        }
                        Some(tool_calls)
                    } else {
                        None
                    };
                    
                    return std::task::Poll::Ready(Some(Ok(ChatStreamItem {
                        content: String::new(),
                        tool_calls: final_tool_calls,
                        done: true,
                        usage: self.usage.clone(),
                    })));
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}