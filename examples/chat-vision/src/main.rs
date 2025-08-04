use futures_util::StreamExt;
use unified_ai::{Message, UnifiedAI};
use std::io::{self, Write};
use std::env;

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

                    // Filter to show only vision models (GPT-4 variants that support vision)
                    let vision_models: Vec<_> = models.into_iter()
                        .filter(|m| m.id.contains("gpt-4") && 
                            (m.id.contains("vision") || m.id.contains("gpt-4o")))
                        .collect();

                    if vision_models.is_empty() {
                        println!("No vision models found, showing all GPT-4 models:");
                        // Fallback to all GPT-4 models if no specific vision models found
                        let temp_client = UnifiedAI::openai(api_key.clone(), "temp".to_string());
                        let all_models = temp_client.get_available_models().await?;
                        let gpt4_models: Vec<_> = all_models.into_iter()
                            .filter(|m| m.id.contains("gpt-4") || m.id.contains("o1"))
                            .collect();
                        
                        if gpt4_models.is_empty() {
                            return Err("No GPT-4 models available".into());
                        }

                        println!("\nAvailable OpenAI models:");
                        for (i, model) in gpt4_models.iter().enumerate() {
                            println!("{}. {} ({})", i + 1, model.name, model.id);
                        }

                        print!("Select model (1-{}): ", gpt4_models.len());
                        io::stdout().flush()?;

                        let mut model_input = String::new();
                        io::stdin().read_line(&mut model_input)?;
                        let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                        if model_choice == 0 || model_choice > gpt4_models.len() {
                            return Err("Invalid model selection".into());
                        }

                        let selected_model = &gpt4_models[model_choice - 1];
                        println!("Selected: {}", selected_model.name);

                        Ok(UnifiedAI::openai(api_key, selected_model.id.clone()))
                    } else {
                        println!("\nAvailable OpenAI vision models:");
                        for (i, model) in vision_models.iter().enumerate() {
                            println!("{}. {} ({})", i + 1, model.name, model.id);
                        }

                        print!("Select model (1-{}): ", vision_models.len());
                        io::stdout().flush()?;

                        let mut model_input = String::new();
                        io::stdin().read_line(&mut model_input)?;
                        let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                        if model_choice == 0 || model_choice > vision_models.len() {
                            return Err("Invalid model selection".into());
                        }

                        let selected_model = &vision_models[model_choice - 1];
                        println!("Selected: {}", selected_model.name);

                        Ok(UnifiedAI::openai(api_key, selected_model.id.clone()))
                    }
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

                    // Filter to show vision-capable models 
                    let vision_models: Vec<_> = models.into_iter()
                        .filter(|m| {
                            let id_lower = m.id.to_lowercase();
                            (id_lower.contains("gpt-4") && (id_lower.contains("vision") || id_lower.contains("gpt-4o"))) ||
                            id_lower.contains("claude") ||
                            id_lower.contains("gemini") ||
                            m.id == "custom"
                        })
                        .collect();

                    if vision_models.is_empty() {
                        return Err("No vision-capable models available".into());
                    }

                    println!("\nAvailable OpenRouter vision models:");
                    for (i, model) in vision_models.iter().enumerate() {
                        println!("{}. {} ({})", i + 1, model.name, model.id);
                    }

                    // Handle custom model input
                    if vision_models.iter().any(|m| m.id == "custom") {
                        println!("\nNote: Select 'Custom Model' to manually enter any OpenRouter vision model ID");
                    }

                    print!("Select model (1-{}): ", vision_models.len());
                    io::stdout().flush()?;

                    let mut model_input = String::new();
                    io::stdin().read_line(&mut model_input)?;
                    let model_choice: usize = model_input.trim().parse().map_err(|_| "Invalid number")?;

                    if model_choice == 0 || model_choice > vision_models.len() {
                        return Err("Invalid model selection".into());
                    }

                    let selected_model = &vision_models[model_choice - 1];
                    
                    let final_model_id = if selected_model.id == "custom" {
                        print!("Enter OpenRouter vision model ID (e.g., anthropic/claude-sonnet-4): ");
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