use futures_util::StreamExt;
use unified_ai::{Message, UnifiedAI};
use std::io::{self, Write};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Chat Vision Example - Unified AI Library");
        println!("\nUsage:");
        println!("  chat-vision <image_path>           - Analyze image and start chat");
        println!("  chat-vision <image_path> <model>   - Use specific model");
        return Ok(());
    }

    let image_path = &args[1];
    let model = if args.len() > 2 {
        args[2].clone()
    } else {
        "qwen2.5vl:7b".to_string()
    };

    println!("Chat Vision Example - Unified AI Library");
    println!("Using model: {}", model);
    println!("Analyzing image: {}\n", image_path);

    // Create client with vision model
    let client = UnifiedAI::ollama(
        "http://localhost:11434".to_string(),
        model,
    );

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

    while let Some(item) = stream.next().await {
        let item = item.map_err(|e| format!("Stream error: {}", e))?;
        
        if !item.content.is_empty() {
            print!("{}", item.content);
            io::stdout().flush()?;
            full_response.push_str(&item.content);
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
        tool_calls: None,
    });

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

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            
            if !item.content.is_empty() {
                print!("{}", item.content);
                io::stdout().flush()?;
                full_response.push_str(&item.content);
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
            tool_calls: None,
        });

        println!();
    }

    Ok(())
}