use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterRequest {
    pub model: String,
    pub messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenRouterStreamOptions>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterStreamOptions {
    pub include_usage: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenRouterFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenRouterFunctionCall>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenRouterChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenRouterUsage>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterChoice {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterContentItem {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<OpenRouterImageUrl>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_parameters: Option<Vec<String>>,
    // We only need id and name, but include the rest as serde_json::Value to avoid parsing errors
    #[serde(flatten)]
    pub _extra: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterModelsResponse {
    pub data: Vec<OpenRouterModel>,
}