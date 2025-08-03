use futures_util::{Stream, StreamExt};
use reqwest::Client;
use std::error::Error;
use std::pin::Pin;

use crate::core::{Message, ToolCall, ChatStreamItem, Tool};
use super::types::*;

pub struct AnthropicClient {
    client: Client,
    api_key: String,
    pub model: String,
    tools: Vec<Tool>,
}

impl AnthropicClient {
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
        false // Anthropic has native tool support
    }

    pub fn set_debug_mode(&mut self, _debug: bool) {
        // Anthropic debug mode not yet implemented or planned
    }

    pub fn debug_mode(&self) -> bool {
        false
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        Ok(true) // Anthropic Claude models support native tool calling
    }

    fn convert_to_anthropic_message(&self, message: &Message) -> AnthropicMessage {
        let mut content_blocks = vec![ContentBlock::Text {
            text: message.content.clone(),
        }];

        // Add images if present
        if let Some(images) = &message.images {
            for image_data in images {
                content_blocks.insert(0, ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/jpeg".to_string(), 
                        data: image_data.clone(),
                    },
                });
            }
        }

        // Add tool calls if present
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                content_blocks.push(ContentBlock::ToolUse {
                    id: format!("call_{}", "generated_id"),
                    name: tool_call.function.name.clone(),
                    input: tool_call.function.arguments.clone(),
                });
            }
        }

        AnthropicMessage {
            role: message.role.clone(),
            content: content_blocks,
        }
    }

    fn convert_tools_to_anthropic(&self) -> Vec<AnthropicTool> {
        self.tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.parameters.clone(),
            })
            .collect()
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|msg| self.convert_to_anthropic_message(msg))
            .collect();

        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages: anthropic_messages,
            system: None,
            temperature: None,
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.convert_tools_to_anthropic())
            },
            stream: Some(true),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Anthropic API error: {}", error_text).into());
        }

        let stream = response.bytes_stream();
        
        let processed_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let lines = chunk.split(|&b| b == b'\n');
                    let mut results = Vec::new();

                    for line in lines {
                        if line.is_empty() {
                            continue;
                        }

                        // Skip "data: " prefix from SSE
                        let line_str = String::from_utf8_lossy(line);
                        if line_str.starts_with("data: ") {
                            let json_str = &line_str[6..];
                            if json_str.trim() == "[DONE]" {
                                results.push(Ok(ChatStreamItem {
                                    content: String::new(),
                                    tool_calls: None,
                                    done: true,
                                }));
                                continue;
                            }

                            match serde_json::from_str::<StreamingEvent>(json_str) {
                                Ok(event) => {
                                    match event {
                                        StreamingEvent::ContentBlockDelta { delta, .. } => {
                                            match delta {
                                                Delta::TextDelta { text } => {
                                                    results.push(Ok(ChatStreamItem {
                                                        content: text,
                                                        tool_calls: None,
                                                        done: false,
                                                    }));
                                                }
                                                Delta::InputJsonDelta { .. } => {
                                                    // Handle tool input streaming if needed
                                                }
                                            }
                                        }
                                        StreamingEvent::MessageStop => {
                                            results.push(Ok(ChatStreamItem {
                                                content: String::new(),
                                                tool_calls: None,
                                                done: true,
                                            }));
                                        }
                                        StreamingEvent::ContentBlockStart { content_block, .. } => {
                                            if let ContentBlock::ToolUse { id: _, name, input } = content_block {
                                                let tool_call = ToolCall {
                                                    function: crate::core::Function {
                                                        name,
                                                        arguments: input,
                                                    },
                                                };
                                                results.push(Ok(ChatStreamItem {
                                                    content: String::new(),
                                                    tool_calls: Some(vec![tool_call]),
                                                    done: false,
                                                }));
                                            }
                                        }
                                        StreamingEvent::Ping => {
                                            // Ignore ping events
                                        }
                                        _ => {
                                            // Handle other event types as needed
                                        }
                                    }
                                }
                                Err(_e) => {
                                    // Ignore parsing errors - they're often due to partial JSON chunks
                                    // which is normal in streaming responses
                                }
                            }
                        }
                    }

                    Ok(results)
                }
                Err(e) => Err(vec![Err(e.to_string())])
            }
        });

        let flattened_stream = processed_stream
            .map(|result| match result {
                Ok(items) => futures_util::stream::iter(items),
                Err(errors) => futures_util::stream::iter(errors),
            })
            .flatten();

        Ok(Box::pin(flattened_stream))
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
                
                tool_responses.push(Message {
                    role: "user".to_string(),
                    content: result,
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }

    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        // Anthropic doesn't need fallback processing
        (content.to_string(), None)
    }
}