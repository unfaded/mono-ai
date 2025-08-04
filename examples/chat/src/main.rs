use futures_util::StreamExt;
use unified_ai::{Message, UnifiedAI};
use unified_ai_macros::tool;
use std::io::{self, Write};
use colored::*;

#[tool]
/// Get the current weather for a given location
fn get_weather(location: String) -> String {
    format!("Weather in {}: 72°F and sunny", location)
}

#[tool]
/// Generate a secure password with specified length
fn generate_password(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789)(*&^%$#@!~";
    
    let mut rng = rand::rng();
    (0..length)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Unified AI Rust Library");
    println!("This demonstrates streaming chat with optional tool calling");

    // Provider selection
    let mut client = select_provider().await?;

    // the rest of the code below works the same regardless of provider
    
    // Add tools (optional) - just comment these out for basic chat
    client.add_tool(get_weather_tool()).await?;
    client.add_tool(generate_password_tool()).await?;

    // Show fallback mode status
    if client.is_fallback_mode().await {
        println!("Using fallback mode for tool calling (model doesn't support native tools)");
    } else {
        println!("Using native tool calling support");
    }

    let mut messages = Vec::new();

    println!("Type your messages (or quit to exit):");

    loop {
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" || input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
            images: None,
            tool_calls: None,
        });

        print!("{}: ", client.model());
        io::stdout().flush()?;

        let mut stream = client.send_chat_request(&messages).await?;
        let mut full_response = String::new();
        let mut tool_calls = None;

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
                break;
            }
        }

        // Add assistant response with tool calls to conversation
        messages.push(Message {
            role: "assistant".to_string(),
            content: full_response,
            images: None,
            tool_calls: tool_calls.clone(), // Include tool calls in the conversation history
        });

        // Handle tool calls
        if let Some(ref tc) = tool_calls {
            // Tool execution status (remove these prints for silent operation)
            for tool_call in tc {
                println!("\n{}", format!("Using {} tool...", tool_call.function.name).truecolor(169, 169, 169));
            }
            
            let tool_responses = client.handle_tool_calls(tc.clone()).await;
            
            // Show tool results
            for (tool_call, response) in tc.iter().zip(tool_responses.iter()) {
                // Extract clean result from encoded format for display
                let clean_result = if response.content.starts_with("TOOL_RESULT:") {
                    // Parse "TOOL_RESULT:tool_id:actual_result" and extract actual_result
                    let parts: Vec<&str> = response.content.splitn(3, ':').collect();
                    if parts.len() == 3 {
                        parts[2]
                    } else {
                        &response.content
                    }
                } else {
                    &response.content
                };
                println!("{}", format!("{} called, result: {}", tool_call.function.name, clean_result).green());
            }
            
            messages.extend(tool_responses);

            // Continue conversation after tool execution  
            print!("{}: ", client.model());
            io::stdout().flush()?;
            let mut tool_stream = client.send_chat_request(&messages).await?;
            let mut final_response = String::new();
            while let Some(item) = tool_stream.next().await {
                let item = item.map_err(|e| format!("Stream error: {}", e))?;
                if !item.content.is_empty() {
                    print!("{}", item.content);
                    io::stdout().flush()?;
                    final_response.push_str(&item.content);
                }
                if item.done {
                    break;
                }
            }
            
            // Add the final assistant response to conversation
            messages.push(Message {
                role: "assistant".to_string(),
                content: final_response,
                images: None,
                tool_calls: None,
            });
        }

        println!();
    }

    Ok(())
}

async fn select_provider() -> Result<UnifiedAI, Box<dyn std::error::Error>> {
    println!("Select AI Provider:");
    println!("1. Ollama (local)");
    println!("2. Anthropic (cloud)");
    println!("3. OpenAI (cloud)");
    println!("4. OpenRouter (cloud)");
    print!("Enter choice (1-4): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim();

    match choice {
        "1" => {
            // Ollama provider
            println!("\nConnecting to Ollama...");
            let temp_client = UnifiedAI::ollama("http://localhost:11434".to_string(), "temp".to_string());
            
            // Get available models
            match temp_client.list_local_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No models available. Please pull a model first using 'ollama pull <model_name>'");
                        return Err("No models available".into());
                    }

                    println!("\nAvailable local models:");
                    for (i, model) in models.iter().enumerate() {
                        println!("{}. {} ({:.1} GB)", i + 1, model.name, model.size as f64 / 1_073_741_824.0);
                    }

                    print!("Select model (1-{}): ", models.len());
                    io::stdout().flush()?;

                    let mut model_input = String::new();
                    io::stdin().read_line(&mut model_input)?;
                    let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                    if model_choice == 0 || model_choice > models.len() {
                        return Err("Invalid model selection".into());
                    }

                    let selected_model = &models[model_choice - 1];
                    println!("\nSelected: {}", selected_model.name);

                    Ok(UnifiedAI::ollama("http://localhost:11434".to_string(), selected_model.name.clone()))
                }
                Err(e) => {
                    println!("Failed to connect to Ollama: {}", e);
                    println!("Make sure Ollama is running on http://localhost:11434");
                    Err(e)
                }
            }
        }
        "2" => {
            let api_key = match std::env::var("ANTHROPIC_API_KEY") {
                Ok(key) => {
                    println!("Using Anthropic API key from environment variable");
                    key
                }
                Err(_) => {
                    print!("Enter Anthropic API key: ");
                    io::stdout().flush()?;
                    
                    let mut input_key = String::new();
                    io::stdin().read_line(&mut input_key)?;
                    let input_key = input_key.trim().to_string();

                    if input_key.is_empty() {
                        return Err("API key cannot be empty".into());
                    }
                    input_key
                }
            };

            println!("\nFetching available models...");
            let temp_client = UnifiedAI::anthropic(api_key.clone(), "temp".to_string());
            
            match temp_client.get_available_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        return Err("No models available".into());
                    }

                    println!("\nAvailable Anthropic models:");
                    for (i, model) in models.iter().enumerate() {
                        println!("{}. {} ({})", i + 1, model.name, model.id);
                    }

                    print!("Select model (1-{}): ", models.len());
                    io::stdout().flush()?;

                    let mut model_input = String::new();
                    io::stdin().read_line(&mut model_input)?;
                    let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                    if model_choice == 0 || model_choice > models.len() {
                        return Err("Invalid model selection".into());
                    }

                    let selected_model = &models[model_choice - 1];
                    println!("\nSelected: {}", selected_model.name);

                    Ok(UnifiedAI::anthropic(api_key, selected_model.id.clone()))
                }
                Err(e) => {
                    println!("Failed to fetch Anthropic models: {}", e);
                    println!("Please check your API key and internet connection");
                    Err(e)
                }
            }
        }
        "3" => {
            // OpenAI provider - try environment variable first
            let api_key = match std::env::var("OPENAI_API_KEY") {
                Ok(key) => {
                    println!("Using OpenAI API key from environment variable");
                    key
                }
                Err(_) => {
                    print!("Enter OpenAI API key: ");
                    io::stdout().flush()?;
                    
                    let mut input_key = String::new();
                    io::stdin().read_line(&mut input_key)?;
                    let input_key = input_key.trim().to_string();

                    if input_key.is_empty() {
                        return Err("API key cannot be empty".into());
                    }
                    input_key
                }
            };

            println!("\nFetching available models...");
            let temp_client = UnifiedAI::openai(api_key.clone(), "temp".to_string());
            
            match temp_client.get_available_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        return Err("No models available".into());
                    }

                    // Filter to show only chat models (exclude embeddings, etc.)
                    let chat_models: Vec<_> = models.into_iter()
                        .filter(|m| m.id.contains("gpt") || m.id.contains("o1"))
                        .collect();

                    if chat_models.is_empty() {
                        return Err("No chat models available".into());
                    }

                    println!("\nAvailable OpenAI models:");
                    for (i, model) in chat_models.iter().enumerate() {
                        println!("{}. {} ({})", i + 1, model.name, model.id);
                    }

                    print!("Select model (1-{}): ", chat_models.len());
                    io::stdout().flush()?;

                    let mut model_input = String::new();
                    io::stdin().read_line(&mut model_input)?;
                    let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                    if model_choice == 0 || model_choice > chat_models.len() {
                        return Err("Invalid model selection".into());
                    }

                    let selected_model = &chat_models[model_choice - 1];
                    println!("Selected: {}", selected_model.name);

                    Ok(UnifiedAI::openai(api_key, selected_model.id.clone()))
                }
                Err(e) => {
                    println!("Failed to fetch OpenAI models: {}", e);
                    println!("Please check your API key and internet connection");
                    Err(e)
                }
            }
        }
        "4" => {
            // OpenRouter provider - try environment variable first
            let api_key = match std::env::var("OPENROUTER_API_KEY") {
                Ok(key) => {
                    println!("Using OpenRouter API key from environment variable");
                    key
                }
                Err(_) => {
                    print!("Enter OpenRouter API key: ");
                    io::stdout().flush()?;
                    
                    let mut input_key = String::new();
                    io::stdin().read_line(&mut input_key)?;
                    let input_key = input_key.trim().to_string();

                    if input_key.is_empty() {
                        return Err("API key cannot be empty".into());
                    }
                    input_key
                }
            };

            println!("\nFetching available models...");
            let temp_client = UnifiedAI::openrouter(api_key.clone(), "temp".to_string());
            
            match temp_client.get_available_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        return Err("No models available".into());
                    }

                    println!("\nAvailable OpenRouter models:");
                    for (i, model) in models.iter().enumerate() {
                        println!("{}. {} ({})", i + 1, model.name, model.id);
                    }

                    // Handle custom model input
                    if models.iter().any(|m| m.id == "custom") {
                        println!("\nNote: Select Custom Model to manually enter any OpenRouter model ID");
                    }

                    print!("Select model (1-{}): ", models.len());
                    io::stdout().flush()?;

                    let mut model_input = String::new();
                    io::stdin().read_line(&mut model_input)?;
                    let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                    if model_choice == 0 || model_choice > models.len() {
                        return Err("Invalid model selection".into());
                    }

                    let selected_model = &models[model_choice - 1];
                    
                    let final_model_id = if selected_model.id == "custom" {
                        print!("Enter OpenRouter model ID (e.g., anthropic/claude-sonnet-4): ");
                        io::stdout().flush()?;
                        let mut custom_model = String::new();
                        io::stdin().read_line(&mut custom_model)?;
                        let custom_model = custom_model.trim().to_string();
                        if custom_model.is_empty() {
                            return Err("Model ID cannot be empty".into());
                        }
                        println!("Selected custom model: {}", custom_model);
                        custom_model
                    } else {
                        println!("Selected: {}", selected_model.name);
                        selected_model.id.clone()
                    };

                    Ok(UnifiedAI::openrouter(api_key, final_model_id))
                }
                Err(e) => {
                    println!("Failed to fetch OpenRouter models: {}", e);
                    println!("Please check your API key and internet connection");
                    Err(e)
                }
            }
        }
        _ => {
            println!("Invalid choice. Exiting.");
            Err("Invalid provider selection".into())
        }
    }
}