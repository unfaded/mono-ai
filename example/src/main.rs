use futures_util::StreamExt;
use ollama_rust::{Message, OllamaClient, OllamaOptions};
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

/// Get system time and date
#[tool]
pub fn get_current_time() -> String {
    println!("[ get_current_time ]");
    
    let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string();
    
    format!("current_time: {}", current_time)
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

                    let mut stream = client.pull_model_stream(&args[2]).await?;
                    while let Some(progress) = stream.next().await {
                        let progress = progress.map_err(|e| format!("Stream error: {}", e))?;
                        if let (Some(completed), Some(total)) = (progress.completed, progress.total)
                        {
                            let percentage = (completed as f64 / total as f64) * 100.0;
                            println!(
                                "{} - {:.1}% ({}/{})",
                                progress.status, percentage, completed, total
                            );
                        } else {
                            println!("{}", progress.status);
                        }
                    }
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
            "generate" => {
                if args.len() > 2 {
                    let model_name = &args[2];
                    let prompt = if args.len() > 3 {
                        args[3..].join(" ")
                    } else {
                        println!("Enter your prompt:");
                        let mut input = String::new();
                        io::stdin().read_line(&mut input)?;
                        input.trim().to_string()
                    };

                    let client = OllamaClient::new(
                        "http://localhost:11434".to_string(),
                        model_name.to_string(),
                    );

                    println!("Generating response\n");
                    let mut stream = client.generate_stream(&prompt).await?;
                    
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        print!("{}", chunk);
                        io::stdout().flush()?;
                    }
                    println!();
                } else {
                    println!("Usage: generate <model_name> [prompt]");
                }
            }
            _ => {
                let client =
                    OllamaClient::new("http://localhost:11434".to_string(), args[1].to_string());
                chat_stream(client).await?
            }
        }
    } else {
        println!("Usage:");
        println!("  cargo run <model_name>           - Interactive chat");
        println!("  cargo run list                   - List local models");
        println!("  cargo run pull <model_name>      - Pull a model");
        println!("  cargo run image <model> <path>   - Analyze image");
        println!("  cargo run info <model_name>      - Show model info");
        println!("  cargo run generate <model> [prompt] - Generate text");
        println!("  cargo run params <model> [prompt]   - Demo parameter effects");
        println!("\nDefaulting to interactive chat with qwen3:8b");
        
        let client = OllamaClient::new(
            "http://localhost:11434".to_string(),
            "qwen3:8b".to_string(),
        );
        chat_stream(client).await?;
    }

    Ok(())
}

async fn chat_stream(mut client: OllamaClient) -> Result<(), Box<dyn std::error::Error>> {
    client.add_tool(get_weather_tool());
    client.add_tool(generate_secure_password_tool());
    client.add_tool(get_current_time_tool());

    let mut messages: Vec<Message> = Vec::new();

    println!("ollama-rust example - type 'exit' to quit");
    println!("Available tools: get_weather, generate_secure_password, get_current_time");

    loop {
        print!("Message: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.is_empty() {
            continue;
        }

        if user_input == "exit" {
            break;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
            images: None,
            tool_calls: None,
        });

        print!("{}: ", client.model);
        io::stdout().flush()?;

        let mut stream = client.send_chat_request_stream(&messages).await?;
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ollama_rust::ToolCall>> = None;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            if !item.content.is_empty() {
                print!("{}", item.content);
                io::stdout().flush()?;
                full_response.push_str(&item.content);
            }
            if let Some(tc) = item.tool_calls {
                tool_calls = Some(tc);
            }
            if item.done {
                println!();
                break;
            }
        }

        messages.push(Message {
            role: "assistant".to_string(),
            content: full_response,
            images: None,
            tool_calls: tool_calls.clone(),
        });

        if let Some(tool_calls) = tool_calls {
            let tool_responses = client.handle_tool_calls(tool_calls);
            messages.extend(tool_responses);

            print!("{}: ", client.model);
            io::stdout().flush()?;

            // Handle tool response streaming
            let mut tool_stream = client.send_chat_request_stream(&messages).await?;
            let mut tool_response = String::new();

            while let Some(item) = tool_stream.next().await {
                let item = item.map_err(|e| format!("Stream error: {}", e))?;
                if !item.content.is_empty() {
                    print!("{}", item.content);
                    io::stdout().flush()?;
                    tool_response.push_str(&item.content);
                }
                if item.done {
                    println!();
                    break;
                }
            }

            messages.push(Message {
                role: "assistant".to_string(),
                content: tool_response,
                images: None,
                tool_calls: None,
            });
        }
    }

    Ok(())
}
