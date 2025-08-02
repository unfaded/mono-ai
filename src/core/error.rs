use std::fmt;

#[derive(Debug)]
pub enum AIRequestError {
    Network(reqwest::Error),
    Json(serde_json::Error),
    IO(std::io::Error),
    Other(String),
}

impl fmt::Display for AIRequestError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AIRequestError::Network(e) => write!(f, "Network error: {}", e),
            AIRequestError::Json(e) => write!(f, "JSON error: {}", e),
            AIRequestError::IO(e) => write!(f, "IO error: {}", e),
            AIRequestError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for AIRequestError {}

impl From<reqwest::Error> for AIRequestError {
    fn from(err: reqwest::Error) -> Self {
        AIRequestError::Network(err)
    }
}

impl From<serde_json::Error> for AIRequestError {
    fn from(err: serde_json::Error) -> Self {
        AIRequestError::Json(err)
    }
}

impl From<std::io::Error> for AIRequestError {
    fn from(err: std::io::Error) -> Self {
        AIRequestError::IO(err)
    }
}