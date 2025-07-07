use ollama_rust::{Message, OllamaClient};
use ollama_rust_macros::tool;
use rand::Rng;
use std::env;
use std::io::{self, Write};

#[tool]
/// Get the current weather in a given location.
fn get_weather(location: String) -> String {
    println!("[ get_weather ] location: {}", location);

    format!("The weather in {} is sunny.", location)
}

/// Generates a secure password with a given length
/// Put the length to 0 if the user doesn't specify a length, this will generate a random password in a secure length
#[tool]
pub fn generate_secure_password(length: usize) -> String {
    println!("[ generate_secure_password ] length: {}", length);

    let effective_length = if length == 0 {
        rand::rng().random_range(16..=32)
    } else {
        length
    };

    let charset =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;:,.<>?";
    let mut rng = rand::rng();
    let mut password = String::with_capacity(effective_length);

    for _ in 0..effective_length {
        let idx = rng.random_range(0..charset.len());
        password.push(charset.chars().nth(idx).unwrap());
    }

    password
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "list" => {
                let client =
                    OllamaClient::new("http://localhost:11434".to_string(), "".to_string());
                let models = client.list_local_models().await?;
                println!("Local models:");
                for model in models {
                    println!("- {}", model.name);
                }
            }
            "pull" => {
                if args.len() > 2 {
                    let client =
                        OllamaClient::new("http://localhost:11434".to_string(), "".to_string());
                    client.pull_model(&args[2]).await?;
                } else {
                    println!("Provide a model name to pull.");
                }
            }
            "image" => {
                if args.len() > 3 {
                    let model_name = &args[2];
                    let image_path = &args[3];
                    let prompt = if args.len() > 4 {
                        Some(args[4].clone())
                    } else {
                        None
                    };

                    let client = OllamaClient::new(
                        "http://localhost:11434".to_string(),
                        model_name.to_string(),
                    );
                    let messages = vec![Message {
                        role: "user".to_string(),
                        content: prompt.unwrap_or_else(|| "What is in this image?".to_string()),
                        images: None,
                        tool_calls: None,
                    }];

                    let (response, _) = client
                        .send_chat_request_with_images(&messages, vec![image_path.to_string()])
                        .await?;
                    println!("Response: {}", response);
                } else {
                    println!("Usage: image <model_name> <image_path> [prompt]");
                }
            }
            "info" => {
                if args.len() > 2 {
                    let client =
                        OllamaClient::new("http://localhost:11434".to_string(), "".to_string());
                    let model_info = client.show_model_info(&args[2]).await?;
                    println!("Model information for {} >", &args[2]);
                    println!("\n>>> Parameters <<<\n\n {}", model_info.parameters);
                    println!("\n>>> License <<<\n\n {}", model_info.license);
                    println!("\n>>> Template <<<\n\n{}", model_info.template);
                } else {
                    println!("Provide a model name.");
                }
            }
            _ => {
                let client =
                    OllamaClient::new("http://localhost:11434".to_string(), args[1].to_string());
                chat(client).await?
            }
        }
    } else {
        let client = OllamaClient::new(
            "http://localhost:11434".to_string(),
            "qwen3:14b".to_string(),
        );
        chat(client).await?;
    }

    Ok(())
}

async fn chat(mut client: OllamaClient) -> Result<(), Box<dyn std::error::Error>> {
    client.add_tool(get_weather_tool());
    client.add_tool(generate_secure_password_tool());

    let mut messages: Vec<Message> = Vec::new();

    loop {
        print!("Message: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.is_empty() {
            continue;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
            images: None,
            tool_calls: None,
        });

        print!("{}: ", client.model);
        io::stdout().flush()?;

        let (response_content, tool_calls) = client.send_chat_request(&messages).await?;

        messages.push(Message {
            role: "assistant".to_string(),
            content: response_content,
            images: None,
            tool_calls: tool_calls.clone(),
        });

        if let Some(tool_calls) = tool_calls {
            let tool_responses = client.handle_tool_calls(tool_calls);
            messages.extend(tool_responses);

            print!("{}: ", client.model);
            io::stdout().flush()?;
            let (response_content, _) = client.send_chat_request(&messages).await?;
            messages.push(Message {
                role: "assistant".to_string(),
                content: response_content,
                images: None,
                tool_calls: None,
            });
        }
    }
}
