use base64::{Engine as _, engine::general_purpose};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use std::io::Write;

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
struct ChatResponse {
    message: Message,
    done: bool,
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
}

impl OllamaClient {
    pub fn new(endpoint: String, model: String) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            model,
            tools: Vec::new(),
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
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
        let mut stream = self
            .client
            .post(&format!("{}/api/pull", self.endpoint))
            .json(&json!({ "name": model_name, "stream": true }))
            .send()
            .await?
            .bytes_stream();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            let lines = chunk.split(|&b| b == b'\n');
            for line in lines {
                if line.is_empty() {
                    continue;
                }
                println!("{}", String::from_utf8_lossy(line));
            }
        }
        Ok(())
    }

    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
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

        self.send_chat_request(&messages_with_images).await
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        let mut request_body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
        });

        if !self.tools.is_empty() {
            let tools_json: Vec<serde_json::Value> =
                self.tools.iter().map(|t| t.to_json()).collect();
            request_body["tools"] = serde_json::Value::Array(tools_json);
        }

        let mut stream = self
            .client
            .post(&format!("{}/api/chat", self.endpoint))
            .json(&request_body)
            .send()
            .await?
            .bytes_stream();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            let lines = chunk.split(|&b| b == b'\n');

            for line in lines {
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_slice::<ChatResponse>(&line) {
                    Ok(chat_response) => {
                        if let Some(tool_call_chunk) = &chat_response.message.tool_calls {
                            tool_calls = Some(tool_call_chunk.to_vec());
                        }
                        if !chat_response.message.content.is_empty() {
                            print!("{}", chat_response.message.content);
                            std::io::stdout().flush()?;
                            full_response.push_str(&chat_response.message.content);
                        }
                        if chat_response.done {
                            println!();
                            return Ok((full_response, tool_calls));
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError parsing response: {}", e);
                        eprintln!("Problematic line: {:?}", String::from_utf8_lossy(&line));
                    }
                }
            }
        }
        Ok((full_response, tool_calls))
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
                tool_responses.push(Message {
                    role: "tool".to_string(),
                    content: result,
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }
}
