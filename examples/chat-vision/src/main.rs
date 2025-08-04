use futures_util::StreamExt;
use unified_ai::{Message, UnifiedAI};
use std::io::{self, Write};
use std::env;

async fn select_provider() -> Result<UnifiedAI, Box<dyn std::error::Error>> {
    println!("\nSelect AI Provider:");
    println!("1. Ollama (local)");
    println!("2. Anthropic (cloud)");
    print!("Enter choice (1-2): ");
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
            // Anthropic provider
            print!("Enter Anthropic API key: ");
            io::stdout().flush()?;
            
            let mut api_key = String::new();
            io::stdin().read_line(&mut api_key)?;
            let api_key = api_key.trim().to_string();

            if api_key.is_empty() {
                return Err("API key cannot be empty".into());
            }

            println!("\nFetching available models...");
            let temp_client = UnifiedAI::anthropic(api_key.clone(), "temp".to_string());
            
            match temp_client.get_available_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        return Err("No models available".into());
                    }

                    println!("\nAvailable Anthropic models:");
                    for (i, model) in models.iter().enumerate() {
                        println!("{}. {} ({})", i + 1, model.display_name, model.id);
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
                    println!("\nSelected: {}", selected_model.display_name);

                    Ok(UnifiedAI::anthropic(api_key, selected_model.id.clone()))
                }
                Err(e) => {
                    println!("Failed to fetch Anthropic models: {}", e);
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Chat Vision Example - Unified AI Library");
        println!("\nUsage:");
        println!("  chat-vision <image_path>   - Analyze image and start chat");
        return Ok(());
    }

    let image_path = &args[1];

    println!("Chat Vision Example - Unified AI Library");
    println!("Analyzing image: {}\n", image_path);

    // Provider selection
    let client = select_provider().await?;

    // Encode image for conversation history
    let encoded_image = client.encode_image_file(image_path).await?;
    
    let mut messages = vec![
        Message {
            role: "user".to_string(),
            content: "What do you see in this image?".to_string(),
            images: Some(vec![encoded_image]),
            tool_calls: None,
        }
    ];

    // Send initial image analysis request
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

    // Add assistant response to conversation
    messages.push(Message {
        role: "assistant".to_string(),
        content: full_response,
        images: None,
        tool_calls: tool_calls.clone(),
    });

    // Handle tool calls if any
    if let Some(ref tc) = tool_calls {
        let tool_responses = client.handle_tool_calls(tc.clone()).await;
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

        // Add assistant response to conversation
        messages.push(Message {
            role: "assistant".to_string(),
            content: full_response,
            images: None,
            tool_calls: tool_calls.clone(),
        });

        // Handle tool calls if any
        if let Some(ref tc) = tool_calls {
            let tool_responses = client.handle_tool_calls(tc.clone()).await;
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