use regex::Regex;
use crate::core::{Tool, ToolCall, Function};

pub struct FallbackToolHandler;

impl FallbackToolHandler {
    pub fn generate_tool_context(tools: &[Tool]) -> String {
        if tools.is_empty() {
            return String::new();
        }

        let mut context = String::from("\n\nYou have access to the following tools. When you need to use a tool, respond with:\n\n<tool_call>\n{\"function\": {\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}}\n</tool_call>\n\nAvailable tools:\n\n");
        
        for tool in tools {
            context.push_str(&format!("{}: {}\n", tool.name, tool.description));
            context.push_str(&format!("Parameters schema: {}\n\n", serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()));
        }
        
        context.push_str("When using tools, wrap the JSON in <tool_call></tool_call> tags as shown above. Don't feel obligated to use tool calls if it doesn't make sense to do so or you weren't instructed. Normally you'll want to present your results to the user after making a tool call, as the user doesn't know the result, unless explicitly told otherwise (example: the user wants many consecutive tool calls).\n");
        context
    }

    pub fn parse_fallback_tool_calls(content: &str) -> Option<Vec<ToolCall>> {
        let xml_regex = Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").ok()?;
        
        let mut all_tool_calls = Vec::new();
        
        for caps in xml_regex.captures_iter(content) {
            if let Some(json_str) = caps.get(1) {
                let json_content = json_str.as_str().trim();
                
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_content) {
                    if let (Some(name), Some(arguments)) = (
                        parsed.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()),
                        parsed.get("function").and_then(|f| f.get("arguments"))
                    ) {
                        all_tool_calls.push(ToolCall {
                            id: None, // Fallback mode doesn't have tool IDs
                            function: Function {
                                name: name.to_string(),
                                arguments: arguments.clone(),
                            }
                        });
                    }
                }
            }
        }
        
        if !all_tool_calls.is_empty() {
            Some(all_tool_calls)
        } else {
            None
        }
    }

    pub fn process_fallback_response(content: &str) -> (String, Option<Vec<ToolCall>>) {
        if let Some(tool_calls) = Self::parse_fallback_tool_calls(content) {
            // Remove the tool call XML from the content
            let xml_regex = Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap();
            let cleaned_content = xml_regex.replace_all(content, "").trim().to_string();
            
            // If cleaned content is empty or very short, indicate tool usage
            let final_content = if cleaned_content.len() < 10 {
                "I'll help you with that.".to_string()
            } else {
                cleaned_content
            };
            
            (final_content, Some(tool_calls))
        } else {
            // Check for incomplete tool calls and remove them
            let incomplete_regex = Regex::new(r"<tool_call>\s*$").unwrap();
            let cleaned_content = incomplete_regex.replace_all(content, "").trim().to_string();
            
            (cleaned_content, None)
        }
    }
}