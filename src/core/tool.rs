use serde_json::Value;

pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub function: Box<dyn Fn(serde_json::Value) -> String + Send + Sync>,
}