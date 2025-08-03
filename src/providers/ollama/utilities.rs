pub struct StreamingXmlFilter {
    inside_tool_call: bool,
}

impl StreamingXmlFilter {
    pub fn new() -> Self {
        Self {
            inside_tool_call: false,
        }
    }

    pub fn process_chunk(&mut self, content: &str) -> String {
        if content.is_empty() {
            return content.to_string();
        }

        let mut result = content.to_string();
        
        if content.contains("<tool_call>") {
            self.inside_tool_call = true;
            result = String::new();
        } else if content.contains("</tool_call>") {
            self.inside_tool_call = false;
            result = String::new();
        } else if self.inside_tool_call {
            result = String::new();
        }

        result
    }

    pub fn is_inside_tool_call(&self) -> bool {
        self.inside_tool_call
    }
}