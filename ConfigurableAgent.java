package com.mycompany.myapp.agents;

import dev.langchain4j.model.chat.ChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StringUtils;
import com.mycompany.myapp.tools.ToolRegistry;
import com.mycompany.myapp.tools.ToolSpecification;
import com.mycompany.myapp.tools.Tool;
import com.mycompany.myapp.tools.ToolResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A configurable implementation of the Agent interface that integrates tools with the chat model.
 */
public class ConfigurableAgent implements Agent {
    
    private static final Logger log = LoggerFactory.getLogger(ConfigurableAgent.class);
    
    private final AgentDefinition definition;
    private final ChatModel chatModel;
    private final ToolRegistry toolRegistry;
    private final Map<String, ToolSpecification> toolSpecifications;
    private final Map<String, Tool> tools;
    private final ObjectMapper objectMapper;
    
    // Maximum number of tool execution iterations allowed
    private static final int MAX_TOOL_CALLS = 5;
    
    // Pattern to detect tool calls in AI responses
    private static final Pattern TOOL_CALL_PATTERN = Pattern.compile(
        "TOOL_CALL:\\s*(\\w+)\\s*\\{([^}]+)\\}", Pattern.MULTILINE | Pattern.DOTALL);
    
    public ConfigurableAgent(AgentDefinition definition, ChatModel chatModel, ToolRegistry toolRegistry) {
        this.definition = definition;
        this.chatModel = chatModel;
        this.toolRegistry = toolRegistry;
        this.toolSpecifications = new HashMap<>();
        this.tools = new HashMap<>();
        this.objectMapper = new ObjectMapper();
        
        // Validate and load tool specifications and tools
        if (definition.getTools() != null) {
            for (String toolName : definition.getTools()) {
                try {
                    // Load tool specification
                    ToolSpecification spec = toolRegistry.getToolSpecification(toolName).block();
                    if (spec != null) {
                        toolSpecifications.put(toolName, spec);
                        log.debug("Loaded tool specification for {}: {}", toolName, spec);
                    }
                    
                    // Load tool instance
                    Tool tool = toolRegistry.getTool(toolName).block();
                    if (tool != null) {
                        tools.put(toolName, tool);
                        log.debug("Loaded tool instance for {}", toolName);
                    }
                } catch (Exception e) {
                    log.warn("Failed to load tool {}: {}", toolName, e.getMessage());
                }
            }
        }
        
        log.info("Agent {} initialized with {} tools", definition.getName(), tools.size());
    }
    
    @Override
    public AgentDefinition getDefinition() {
        return definition;
    }
    
    /**
     * Gets the specifications of tools available to this agent.
     *
     * @return map of tool names to their specifications
     */
    public Map<String, ToolSpecification> getToolSpecifications() {
        return toolSpecifications;
    }
    
    @Override
    public String chat(Map<String, Object> context) {
        String systemPrompt = renderSystemPrompt(context);
        String userPrompt = renderUserPrompt(context);
        
        log.debug("System prompt: {}", systemPrompt);
        log.debug("User prompt: {}", userPrompt);
        
        // Create the full prompt with system and user messages
        String fullPrompt = systemPrompt + "\n\nUser: " + userPrompt + "\nAssistant: ";
        
        // Execute chat with potential tool calls
        return executeChat(fullPrompt, context);
    }
    
    /**
     * Executes the chat request, handling tool calls if necessary.
     */
    private String executeChat(String prompt, Map<String, Object> context) {
        int iteration = 0;
        StringBuilder conversationBuilder = new StringBuilder(prompt);
        
        while (iteration < MAX_TOOL_CALLS) {
            try {
                String response = chatModel.chat(conversationBuilder.toString());
                log.debug("AI response: {}", response);
                
                // Check if the response contains tool calls
                Matcher matcher = TOOL_CALL_PATTERN.matcher(response);
                if (matcher.find()) {
                    log.debug("AI requested tool execution");
                    
                    // Extract and execute tool calls
                    StringBuffer responseWithResults = new StringBuffer();
                    do {
                        String toolName = matcher.group(1);
                        String parametersJson = matcher.group(2);
                        
                        String toolResult = executeTool(toolName, parametersJson, context);
                        String replacement = "TOOL_RESULT: " + toolResult;
                        matcher.appendReplacement(responseWithResults, replacement);
                        
                        log.debug("Executed tool {}, result: {}", toolName, toolResult);
                    } while (matcher.find());
                    matcher.appendTail(responseWithResults);
                    
                    // Continue conversation with tool results
                    conversationBuilder.append(responseWithResults.toString()).append("\n\nPlease provide a final response based on the tool results: ");
                    iteration++;
                    continue;
                }
                
                // No tool calls, return the response
                return response;
                
            } catch (Exception e) {
                log.error("Error during chat execution: {}", e.getMessage(), e);
                return "I encountered an error while processing your request. Please try again.";
            }
        }
        
        log.warn("Maximum tool execution iterations ({} calls) reached", MAX_TOOL_CALLS);
        return "I'm sorry, but I couldn't complete your request due to too many tool interactions.";
    }
    
    /**
     * Executes a specific tool with the given arguments.
     */
    private String executeTool(String toolName, String parametersJson, Map<String, Object> context) {
        Tool tool = tools.get(toolName);
        if (tool == null) {
            log.warn("Tool not found: {}", toolName);
            return "Error: Tool not available";
        }
        
        try {
            // Parse JSON parameters
            Map<String, Object> parameters = objectMapper.readValue(
                parametersJson, new TypeReference<Map<String, Object>>() {});
            
            // Validate parameters against tool definition
            if (!tool.validateParameters(parameters)) {
                return "Error: Invalid parameters for tool " + toolName;
            }
            
            // Execute tool
            ToolResult result = tool.execute(parameters, context).block();
            if (result != null && result.isSuccess()) {
                return result.getResult() != null ? result.getResult().toString() : "Tool executed successfully";
            } else {
                return "Error: " + (result != null ? result.getErrorMessage() : "Tool execution failed");
            }
        } catch (Exception e) {
            log.error("Error executing tool {}: {}", toolName, e.getMessage(), e);
            return "Error executing tool: " + e.getMessage();
        }
    }
    
    @Override
    public List<String> streamChat(Map<String, Object> context) {
        // For simplicity, fall back to regular chat for now
        // Streaming with tools requires more complex setup
        String result = chat(context);
        List<String> tokens = new ArrayList<>();
        // Split into word-based tokens for simple streaming simulation
        String[] words = result.split("\\s+");
        for (String word : words) {
            tokens.add(word + " ");
        }
        return tokens;
    }
    
    private String renderSystemPrompt(Map<String, Object> context) {
        Map<String, Object> variables = new HashMap<>(context);
        variables.put("agent_name", definition.getName());
        variables.put("agent_description", definition.getDescription());
        variables.put("current_datetime", LocalDateTime.now());
        
        if (definition.getConfiguration() != null) {
            variables.putAll(definition.getConfiguration());
        }
        
        // Add available tools information to system prompt
        if (!toolSpecifications.isEmpty()) {
            StringBuilder toolsInfo = new StringBuilder("\n\nYou have access to the following tools. To use a tool, respond with TOOL_CALL: toolName {\"param1\": \"value1\", \"param2\": \"value2\"}\n\n");
            for (Map.Entry<String, ToolSpecification> entry : toolSpecifications.entrySet()) {
                ToolSpecification spec = entry.getValue();
                toolsInfo.append("- ").append(spec.getName())
                    .append(": ").append(spec.getDescription());
                
                if (spec.getParameters() != null && !spec.getParameters().isEmpty()) {
                    toolsInfo.append("\n  Parameters: ");
                    for (ToolSpecification.ToolParameter param : spec.getParameters()) {
                        toolsInfo.append(param.getName()).append(" (").append(param.getType()).append(") ");
                        if (param.isRequired()) {
                            toolsInfo.append("[required] ");
                        }
                        if (StringUtils.hasText(param.getDescription())) {
                            toolsInfo.append("- ").append(param.getDescription());
                        }
                        toolsInfo.append("; ");
                    }
                }
                toolsInfo.append("\n");
            }
            variables.put("available_tools", toolsInfo.toString());
        } else {
            variables.put("available_tools", "");
        }
        
        String template = definition.getPromptTemplates().get("system");
        if (template == null) {
            template = "You are a helpful AI assistant named {{agent_name}}. {{agent_description}}{{available_tools}}";
        }
        
        return renderTemplate(template, variables);
    }
    
    private String renderUserPrompt(Map<String, Object> context) {
        Map<String, Object> variables = new HashMap<>(context);
        if (!variables.containsKey("knowledge_context")) {
            variables.put("knowledge_context", "");
        }
        
        String template = definition.getPromptTemplates().get("user");
        if (template == null) {
            template = "{{user_message}}\n{{#knowledge_context}}Using this relevant knowledge:\n{{knowledge_context}}{{/knowledge_context}}";
        }
        
        return renderTemplate(template, variables);
    }
    
    private String renderTemplate(String template, Map<String, Object> variables) {
        String result = template;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String placeholder = "{{" + entry.getKey() + "}}";
            if (entry.getValue() != null) {
                result = result.replace(placeholder, entry.getValue().toString());
            } else {
                result = result.replace(placeholder, "");
            }
        }
        return result;
    }
} 