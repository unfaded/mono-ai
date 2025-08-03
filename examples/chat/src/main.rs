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

    // Create client - Choose your provider:
    
    // Option 1: Ollama (local)
    //let mut client = UnifiedAI::ollama(
    //    "http://localhost:11434".to_string(),
    //    "qwen3-coder:30b".to_string(),
    //);

    // For cloud providers: You can hardcode keys instead of using environment variables if preferred
    // Option 2: Anthropic Claude (requires API key)
    // let mut client = UnifiedAI::anthropic(
    //     std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
    //     "claude-sonnet-4-20250514".to_string(),
    // );
    
    let mut client = UnifiedAI::anthropic(
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
        "claude-sonnet-4-20250514".to_string(),
    );
    // The rest of the code works exactly the same regardless of provider!

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
                println!("{}", format!("{} called, result: {}", tool_call.function.name, response.content).green());
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