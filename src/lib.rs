use base64::{Engine as _, engine::general_purpose};
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use std::io::Write;
use std::pin::Pin;
use regex::Regex;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub message: Message,
    pub done: bool,
}

#[derive(Debug)]
pub struct ChatStreamItem {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
}

#[derive(Debug)]
pub struct PullProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

#[derive(Deserialize, Debug)]
pub struct Model {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub license: String,
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
}

#[derive(Deserialize, Debug)]
struct ListModelsResponse {
    models: Vec<Model>,
}

#[derive(Serialize, Debug, Default)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gqa: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_vram: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f16_kv: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_all: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mlock: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<i32>,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub function: Box<dyn Fn(serde_json::Value) -> String + Send + Sync>,
}

impl Tool {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

pub struct OllamaClient {
    client: Client,
    pub endpoint: String,
    pub model: String,
    tools: Vec<Tool>,
    fallback_mode: bool,
}

impl OllamaClient {
    pub fn new(endpoint: String, model: String) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            model,
            tools: Vec::new(),
            fallback_mode: false,
        }
    }

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        self.tools.push(tool);
        
        // Only check tool support on the first tool to avoid multiple checks
        if self.tools.len() == 1 {
            let supports_native = self.supports_tool_calls().await?;
            if !supports_native {
                self.fallback_mode = true;
            }
        }
        
        Ok(())
    }

    pub fn is_fallback_mode(&self) -> bool {
        self.fallback_mode
    }

    fn generate_tool_context(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut context = String::from("\n\nYou have access to the following tools. When you need to use a tool, respond with:\n\n<tool_call>\n{\"function\": {\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}}\n</tool_call>\n\nAvailable tools:\n\n");
        
        for tool in &self.tools {
            context.push_str(&format!("{}: {}\n", tool.name, tool.description));
            context.push_str(&format!("Parameters schema: {}\n\n", serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()));
        }
        
        context.push_str("When using tools, wrap the JSON in <tool_call></tool_call> tags as shown above. Don't feel obligated to use tool calls if it doesn't make sense to do so or you weren't instructed. Normally you'll want to present your results to the user after making a tool call, as the user doesn't know the result, unless explicitly told otherwise (example: the user wants many consecutive tool calls).\n");
        context
    }

    fn parse_fallback_tool_calls(&self, content: &str) -> Option<Vec<ToolCall>> {
        let xml_regex = Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").ok()?;
        
        let mut all_tool_calls = Vec::new();
        
        for caps in xml_regex.captures_iter(content) {
            if let Some(json_str) = caps.get(1) {
                let json_content = json_str.as_str().trim();
                
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_content) {
                    if let (Some(name), Some(arguments)) = (
                        parsed.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()),
                        parsed.get("function").and_then(|f| f.get("arguments"))
                    ) {
                        all_tool_calls.push(ToolCall {
                            function: Function {
                                name: name.to_string(),
                                arguments: arguments.clone(),
                            }
                        });
                    }
                }
            }
        }
        
        if !all_tool_calls.is_empty() {
            Some(all_tool_calls)
        } else {
            None
        }
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        let model_info = self.show_model_info(&self.model).await?;
        
        // The definitive way to check tool support is the presence of {{ .Tools }} in the template
        // All models that support tools use the {{ .Tools }} variable in their prompt template
        let template = &model_info.template;
        let supports_tools = template.contains("{{ .Tools }}") || 
                           template.contains("{{.Tools}}") ||
                           template.contains("{{ .tools }}") ||
                           template.contains("{{.tools}}") ||
                           template.contains("{{- .Tools }}") ||
                           template.contains("{{- .tools }}");
        
        Ok(supports_tools)
    }

    pub async fn list_local_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        let response = self
            .client
            .get(&format!("{}/api/tags", self.endpoint))
            .send()
            .await?
            .json::<ListModelsResponse>()
            .await?;
        Ok(response.models)
    }

    pub async fn show_model_info(&self, model_name: &str) -> Result<ModelInfo, Box<dyn Error>> {
        let response = self
            .client
            .post(&format!("{}/api/show", self.endpoint))
            .json(&json!({ "name": model_name }))
            .send()
            .await?
            .json::<ModelInfo>()
            .await?;
        Ok(response)
    }

    pub async fn pull_model(&self, model_name: &str) -> Result<(), Box<dyn Error>> {
        println!("Pulling model: {}", model_name);
        let mut stream = self.pull_model_stream(model_name).await?;

        while let Some(progress) = stream.next().await {
            let progress = progress.map_err(|e| format!("Stream error: {}", e))?;
            println!("{}", progress.status);
        }
        Ok(())
    }

    pub async fn pull_model_stream(
        &self,
        model_name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullProgress, String>> + Send>>, Box<dyn Error>>
    {
        let stream = self
            .client
            .post(&format!("{}/api/pull", self.endpoint))
            .json(&json!({ "name": model_name, "stream": true }))
            .send()
            .await?
            .bytes_stream();

        let stream = stream.map(
            |item| -> Result<Vec<Result<PullProgress, String>>, Box<dyn Error>> {
                let chunk = item?;
                let lines = chunk.split(|&b| b == b'\n');
                let mut results = Vec::new();

                for line in lines {
                    if line.is_empty() {
                        continue;
                    }

                    let line_str = String::from_utf8_lossy(line);
                    match serde_json::from_str::<serde_json::Value>(&line_str) {
                        Ok(json) => {
                            results.push(Ok(PullProgress {
                                status: json
                                    .get("status")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                digest: json
                                    .get("digest")
                                    .and_then(|s| s.as_str())
                                    .map(|s| s.to_string()),
                                total: json.get("total").and_then(|n| n.as_u64()),
                                completed: json.get("completed").and_then(|n| n.as_u64()),
                            }));
                        }
                        Err(_) => {
                            results.push(Ok(PullProgress {
                                status: line_str.to_string(),
                                digest: None,
                                total: None,
                                completed: None,
                            }));
                        }
                    }
                }

                Ok(results)
            },
        );

        let flattened_stream = stream
            .map(
                |result: Result<Vec<Result<PullProgress, String>>, Box<dyn Error>>| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
    }

    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        self.send_chat_request_with_images_and_options(messages, image_paths, None).await
    }

    pub async fn send_chat_request_with_images_and_options(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
        options: Option<OllamaOptions>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut encoded_images = Vec::new();
        for image_path in image_paths {
            let image_bytes = std::fs::read(image_path)?;
            encoded_images.push(general_purpose::STANDARD.encode(image_bytes));
        }

        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            last_message.images = Some(encoded_images);
        }

        self.send_chat_request_with_options(&messages_with_images, options).await
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        self.send_chat_request_with_options(messages, None).await
    }

    pub async fn send_chat_request_with_options(
        &self,
        messages: &[Message],
        options: Option<OllamaOptions>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut stream = self.send_chat_request_stream_with_options(messages, options).await?;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            if !item.content.is_empty() {
                print!("{}", item.content);
                std::io::stdout().flush()?;
                full_response.push_str(&item.content);
            }
            if let Some(tc) = item.tool_calls {
                tool_calls = Some(tc);
            }
            if item.done {
                println!();
                return Ok((full_response, tool_calls));
            }
        }
        Ok((full_response, tool_calls))
    }

    pub async fn send_chat_request_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>>
    {
        self.send_chat_request_stream_with_options(messages, None).await
    }

    pub async fn send_chat_request_stream_with_options(
        &self,
        messages: &[Message],
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>>
    {
        let mut messages_to_send = messages.to_vec();
        
        // In fallback mode, inject tool context into the system message
        if self.fallback_mode && !self.tools.is_empty() {
            let tool_context = self.generate_tool_context();
            
            // Find existing system message or create one
            if let Some(system_msg) = messages_to_send.iter_mut().find(|msg| msg.role == "system") {
                system_msg.content.push_str(&tool_context);
            } else {
                // Insert system message at the beginning
                messages_to_send.insert(0, Message {
                    role: "system".to_string(),
                    content: format!("You are a helpful assistant.{}", tool_context),
                    images: None,
                    tool_calls: None,
                });
            }
        }

        let mut request_body = json!({
            "model": self.model,
            "messages": messages_to_send,
            "stream": true,
        });

        // Only add tools if not in fallback mode
        if !self.fallback_mode && !self.tools.is_empty() {
            let tools_json: Vec<serde_json::Value> =
                self.tools.iter().map(|t| t.to_json()).collect();
            request_body["tools"] = serde_json::Value::Array(tools_json);
        }

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let stream = self
            .client
            .post(&format!("{}/api/chat", self.endpoint))
            .json(&request_body)
            .send()
            .await?
            .bytes_stream();

        let fallback_mode = self.fallback_mode;
        let stream = stream.map(
            move |item| -> Result<Vec<Result<ChatStreamItem, String>>, Box<dyn Error>> {
                let chunk = item?;
                let lines = chunk.split(|&b| b == b'\n');
                let mut results = Vec::new();

                for line in lines {
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_slice::<ChatResponse>(&line) {
                        Ok(chat_response) => {
                            let tool_calls = chat_response.message.tool_calls.clone();
                            
                            // In fallback mode, try to parse tool calls from content
                            if fallback_mode && tool_calls.is_none() && !chat_response.message.content.is_empty() {
                                // Note: We can't call self.parse_fallback_tool_calls here because of ownership
                                // This will be handled in the client code after collecting the full response
                            }
                            
                            results.push(Ok(ChatStreamItem {
                                content: chat_response.message.content.clone(),
                                tool_calls,
                                done: chat_response.done,
                            }));
                        }
                        Err(e) => {
                            eprintln!("\nError parsing response: {}", e);
                            eprintln!("Problematic line: {:?}", String::from_utf8_lossy(&line));
                        }
                    }
                }

                Ok(results)
            },
        );

        let flattened_stream = stream
            .map(
                |result: Result<Vec<Result<ChatStreamItem, String>>, Box<dyn Error>>| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
    }

    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn Error>> {
        self.generate_with_options(prompt, None).await
    }

    pub async fn generate_with_options(
        &self,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<String, Box<dyn Error>> {
        let mut request_body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let response = self
            .client
            .post(&format!("{}/api/generate", self.endpoint))
            .json(&request_body)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        Ok(response_json["response"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        self.generate_stream_with_options(prompt, None).await
    }

    pub async fn generate_stream_with_options(
        &self,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        let mut request_body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": true,
        });

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let stream = self
            .client
            .post(&format!("{}/api/generate", self.endpoint))
            .json(&request_body)
            .send()
            .await?
            .bytes_stream();

        let stream = stream.map(
            |item| -> Result<Vec<Result<String, String>>, Box<dyn Error>> {
                let chunk = item?;
                let lines = chunk.split(|&b| b == b'\n');
                let mut results = Vec::new();

                for line in lines {
                    if line.is_empty() {
                        continue;
                    }

                    match serde_json::from_slice::<serde_json::Value>(&line) {
                        Ok(json) => {
                            if let Some(response) = json["response"].as_str() {
                                results.push(Ok(response.to_string()));
                            }
                        }
                        Err(e) => {
                            results.push(Err(format!("Parse error: {}", e)));
                        }
                    }
                }

                Ok(results)
            },
        );

        let flattened_stream = stream
            .map(
                |result: Result<Vec<Result<String, String>>, Box<dyn Error>>| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
    }

    pub fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        let mut tool_responses = Vec::new();
        for tool_call in tool_calls {
            if let Some(tool) = self
                .tools
                .iter()
                .find(|t| t.name == tool_call.function.name)
            {
                let result = (tool.function)(tool_call.function.arguments.clone());
                
                // In fallback mode, format tool response as user message with tool context
                let (role, content) = if self.fallback_mode {
                    ("user".to_string(), format!("Tool response from {}: {}", tool_call.function.name, result))
                } else {
                    ("tool".to_string(), result)
                };
                
                tool_responses.push(Message {
                    role,
                    content,
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }

    pub fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        if !self.fallback_mode {
            return (content.to_string(), None);
        }

        if let Some(tool_calls) = self.parse_fallback_tool_calls(content) {
            // Remove the tool call XML from the content
            let xml_regex = Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap();
            let cleaned_content = xml_regex.replace_all(content, "").trim().to_string();
            
            // If cleaned content is empty or very short, indicate tool usage
            let final_content = if cleaned_content.len() < 10 {
                "I'll help you with that.".to_string()
            } else {
                cleaned_content
            };
            
            (final_content, Some(tool_calls))
        } else {
            // Check for incomplete tool calls and remove them
            let incomplete_regex = Regex::new(r"<tool_call>\s*$").unwrap();
            let cleaned_content = incomplete_regex.replace_all(content, "").trim().to_string();
            
            (cleaned_content, None)
        }
    }
}
