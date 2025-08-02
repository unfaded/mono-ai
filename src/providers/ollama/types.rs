use serde::Deserialize;
use crate::core::Message;

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub message: Message,
    pub done: bool,
}

#[derive(Deserialize, Debug)]
pub struct Model {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
}

#[derive(Deserialize, Debug)]
pub struct ListModelsResponse {
    pub models: Vec<Model>,
}