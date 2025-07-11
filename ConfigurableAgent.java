package com.mycompany.myapp.agents;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.mycompany.myapp.tools.Tool;
import com.mycompany.myapp.tools.ToolRegistry;
import com.mycompany.myapp.tools.ToolResult;
import com.mycompany.myapp.tools.ToolSpecification;
import dev.langchain4j.model.chat.ChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.LocalDateTime;
import java.util.*;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A configurable implementation of the Agent interface that integrates tools with the chat model
 * and is able to execute multiple independent tool calls in parallel.
 * <p>Response format contract:</p>
 * <pre>
 *   TOOL_CALL:   <toolName> {"param":"value"}
 *   TOOL_RESULT: <toolName> {"success":true|false, "result":<any>, "error":<msg>}
 * </pre>
 * The assistant therefore always echoes the <strong>same tool name</strong> back with its result JSON so
 * the LLM can unambiguously correlate each request/response pair.
 */
public class ConfigurableAgent implements Agent {

    private static final Logger log = LoggerFactory.getLogger(ConfigurableAgent.class);

    // ---------- Static configuration ----------------------------------------------------------
    private static final int MAX_TOOL_CALLS = 5;
    private static final Pattern TOOL_CALL_PATTERN = Pattern.compile(
        "TOOL_CALL:\\s*(\\w+)\\s*\\{([^}]+)\\}", Pattern.MULTILINE | Pattern.DOTALL);

    // ---------- Immutable collaborators -------------------------------------------------------
    private final AgentDefinition definition;
    private final ChatModel      chatModel;
    private final ToolRegistry   toolRegistry;

    // ---------- Mutable state ----------------------------------------------------------------
    private final Map<String, ToolSpecification> toolSpecifications = new HashMap<>();
    private final Map<String, Tool>              tools              = new HashMap<>();

    private final ObjectMapper objectMapper = new ObjectMapper();

    // ------------------------------------------------------------------------------------------
    public ConfigurableAgent(AgentDefinition definition,
                             ChatModel        chatModel,
                             ToolRegistry     toolRegistry) {
        this.definition   = definition;
        this.chatModel    = chatModel;
        this.toolRegistry = toolRegistry;

        if (definition.getTools() != null) {
            for (String toolName : definition.getTools()) {
                try {
                    ToolSpecification spec = toolRegistry.getToolSpecification(toolName).block();
                    if (spec != null) {
                        toolSpecifications.put(toolName, spec);
                        log.debug("Loaded tool specification for {}: {}", toolName, spec);
                    }
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

    // ------------------------------------------------------------------------------------------------
    // Public API -------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------
    @Override
    public String chat(Map<String, Object> context) {
        String systemPrompt = renderSystemPrompt(context);
        String userPrompt   = renderUserPrompt(context);

        log.debug("System prompt: {}", systemPrompt);
        log.debug("User prompt: {}", userPrompt);

        String fullPrompt = systemPrompt + "\n\nUser: " + userPrompt + "\nAssistant: ";
        return executeChat(fullPrompt, context);
    }

    @Override
    public List<String> streamChat(Map<String, Object> context) {
        String result = chat(context);
        List<String> tokens = new ArrayList<>();
        for (String w : result.split("\\s+")) tokens.add(w + " ");
        return tokens;
    }

    // ------------------------------------------------------------------------------------------------
    // Core logic -------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------

    private String executeChat(String prompt, Map<String, Object> context) {
        int           iteration    = 0;
        StringBuilder conversation = new StringBuilder(prompt);

        while (iteration < MAX_TOOL_CALLS) {
            try {
                String response = chatModel.chat(conversation.toString());
                log.debug("AI response: {}", response);

                Matcher matcher = TOOL_CALL_PATTERN.matcher(response);
                if (matcher.find()) {
                    // 1. Capture all tool calls
                    List<PendingCall> pending = new ArrayList<>();
                    matcher.reset();
                    while (matcher.find()) {
                        pending.add(new PendingCall(matcher.toMatchResult(), matcher.group(1), matcher.group(2)));
                    }

                    // 2. Parallel execution – results preserved in order
                    List<ToolResult> results = Flux.fromIterable(pending)
                        .flatMapSequential(pc -> executeToolAsync(pc.toolName(), pc.paramsJson(), context)
                            .subscribeOn(Schedulers.boundedElastic()))
                        .collectList()
                        .block();

                    // 3. Replace calls with results (iterate from tail to head to keep indexes)
                    StringBuilder rebuilt = new StringBuilder(response);
                    for (int i = pending.size() - 1; i >= 0; i--) {
                        PendingCall pc = pending.get(i);
                        ToolResult  tr = results.get(i);

                        String json = buildResultJson(pc.toolName(), tr);
                        String replacement = "TOOL_RESULT: " + pc.toolName() + " " + json;

                        rebuilt.replace(pc.match().start(), pc.match().end(), replacement);
                    }

                    conversation.append(rebuilt)
                        .append("\n\nPlease provide a final response based on the tool results: ");
                    iteration++;
                    continue; // restart chat loop with appended convo
                }

                // No tool calls – finished
                return response;

            } catch (Exception e) {
                log.error("Error during chat execution", e);
                return "I encountered an error while processing your request. Please try again.";
            }
        }
        log.warn("Maximum tool execution iterations ({}) reached", MAX_TOOL_CALLS);
        return "I'm sorry, but I couldn't complete your request due to too many tool interactions.";
    }

    /**
     * Execute a tool asynchronously and return a {@link Mono} that completes with its {@link ToolResult}.
     */
    private Mono<ToolResult> executeToolAsync(String toolName, String parametersJson, Map<String, Object> context) {
        Tool tool = tools.get(toolName);
        if (tool == null) {
            log.warn("Tool not found: {}", toolName);
            return Mono.just(ToolResult.error("Tool not available"));
        }

        return Mono.defer(() -> {
                Map<String, Object> params;
                try {
                    params = objectMapper.readValue(parametersJson, new TypeReference<Map<String, Object>>() {});
                } catch (JsonProcessingException e) {
                    return Mono.just(ToolResult.error("Invalid JSON parameters: " + e.getMessage()));
                }

                if (!tool.validateParameters(params)) {
                    return Mono.just(ToolResult.error("Invalid parameters for tool " + toolName));
                }
                return tool.execute(params, context);
            })
            .timeout(java.time.Duration.ofSeconds(60))
            .onErrorResume(e -> {
                log.error("Tool {} execution failed: {}", toolName, e.toString());
                return Mono.just(ToolResult.error(e.getMessage()));
            });
    }

    /**
     * Build the canonical JSON that is echoed back to the LLM after a tool finishes.
     */
    private String buildResultJson(String toolName, ToolResult tr) {
        ObjectNode node = objectMapper.createObjectNode();
        node.put("tool", toolName);
        if (tr == null) {
            node.put("success", false);
            node.put("error", "null ToolResult");
        } else if (tr.isSuccess()) {
            node.put("success", true);
            if (tr.getResult() != null) node.putPOJO("result", tr.getResult());
        } else {
            node.put("success", false);
            node.put("error", tr.getErrorMessage());
        }
        try {
            return objectMapper.writeValueAsString(node);
        } catch (JsonProcessingException e) {
            // This should never fail but fallback if it does
            return "{\"tool\":\"" + toolName + "\",\"success\":false,\"error\":\"JSON encode error\"}";
        }
    }

    // ------------------------------------------------------------------------------------------------
    // Prompt helpers -------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------

    private String renderSystemPrompt(Map<String, Object> context) {
        Map<String, Object> vars = new HashMap<>(context);
        vars.put("agent_name", definition.getName());
        vars.put("agent_description", definition.getDescription());
        vars.put("current_datetime", LocalDateTime.now());
        if (definition.getConfiguration() != null) vars.putAll(definition.getConfiguration());

        // Build tool catalogue for the system prompt
        if (!toolSpecifications.isEmpty()) {
            StringBuilder sb = new StringBuilder("\n\nYou have access to the following tools. To use a tool, respond with TOOL_CALL: toolName {\"param1\": \"value\"}\n\n");
            for (ToolSpecification spec : toolSpecifications.values()) {
                sb.append("- ").append(spec.getName()).append(": ").append(spec.getDescription());
                if (spec.getParameters() != null && !spec.getParameters().isEmpty()) {
                    sb.append("\n  Parameters: ");
                    for (ToolSpecification.ToolParameter p : spec.getParameters()) {
                        sb.append(p.getName()).append(" (").append(p.getType()).append(") ");
                        if (p.isRequired()) sb.append("[required] ");
                        if (StringUtils.hasText(p.getDescription())) sb.append("- ").append(p.getDescription());
                        sb.append("; ");
                    }
                }
                sb.append("\n");
            }
            vars.put("available_tools", sb.toString());
        } else {
            vars.put("available_tools", "");
        }

        String template = definition.getPromptTemplates().get("system");
        if (template == null) template = "You are a helpful AI assistant named {{agent_name}}. {{agent_description}}{{available_tools}}";
        return renderTemplate(template, vars);
    }

    private String renderUserPrompt(Map<String, Object> context) {
        Map<String, Object> vars = new HashMap<>(context);
        vars.putIfAbsent("knowledge_context", "");

        String template = definition.getPromptTemplates().get("user");
        if (template == null) template = "{{user_message}}\n{{#knowledge_context}}Using this relevant knowledge:\n{{knowledge_context}}{{/knowledge_context}}";
        return renderTemplate(template, vars);
    }

    private String renderTemplate(String template, Map<String, Object> vars) {
        String result = template;
        for (Map.Entry<String, Object> e : vars.entrySet()) {
            String placeholder = "{{" + e.getKey() + "}}";
            result = result.replace(placeholder, e.getValue() != null ? e.getValue().toString() : "");
        }
        return result;
    }

    // ------------------------------------------------------------------------------------------------
    // Helper record ----------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------
    private record PendingCall(MatchResult match, String toolName, String paramsJson) {}
}
