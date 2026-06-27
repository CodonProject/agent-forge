import json
from typing import List, Dict, Any, Optional
from agentforge.llm.openai import OpenAICompat
from agentforge.tool import Tool


class Agent:
    def __init__(
        self,
        name: str,
        llm: OpenAICompat,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Tool]] = None,
        model: str = "gpt-4o",
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model
        
        # 将工具列表转化为字典方便查询: {tool_name: tool_instance}
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        
        # 简单内存管理（消息历史）
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def run(self, user_input: str, max_steps: int = 5) -> str:
        """
        同步运行 Agent。如果模型决定调用工具，会自动执行并迭代，直到输出最终文本。
        """
        self.messages.append({"role": "user", "content": user_input})
        
        for step in range(max_steps):
            # 1. 准备工具的 JSON Schema
            tools_schemas = [t.info for t in self.tools.values()] if self.tools else None
            
            # 2. 调用大模型
            response_generator = self.llm.request(
                messages=self.messages,
                tools=tools_schemas,
                model=self.model
            )
            
            # 3. 消费生成器，获取最终状态
            final_status = None
            for response in response_generator:
                if response.is_final:
                    final_status = response.final_status
                    break
            
            if not final_status:
                raise RuntimeError("Failed to get a valid response from LLM.")
            
            # 4. 保存大模型的回复到历史中
            assistant_msg = {"role": "assistant"}
            if final_status.content:
                assistant_msg["content"] = final_status.content
            if final_status.tool_calls:
                assistant_msg["tool_calls"] = final_status.tool_calls
            
            # 防止 OpenAI 接口报错：如果没有 content 也没有 tool_calls，给个空字符串
            if "content" not in assistant_msg and "tool_calls" not in assistant_msg:
                assistant_msg["content"] = ""
                
            self.messages.append(assistant_msg)
            
            # 5. 如果没有工具调用，说明 Agent 思考结束，直接返回结果
            if not final_status.tool_calls:
                return final_status.content
                
            # 6. 执行工具调用
            for tool_call in final_status.tool_calls:
                call_id = tool_call.get("id")
                func_name = tool_call["function"]["name"]
                func_args_str = tool_call["function"]["arguments"]
                
                # 解析参数
                try:
                    args = json.loads(func_args_str) if func_args_str else {}
                except Exception as e:
                    args = {}
                    
                # 寻找工具并执行
                tool = self.tools.get(func_name)
                if tool:
                    try:
                        print(f"[Agent] Executing tool '{func_name}' with args: {args}")
                        result = tool(**args)
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                else:
                    result = f"Error: Tool '{func_name}' not found."
                
                # 将工具执行结果作为 'tool' 角色返回给模型
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": str(result)
                })
                
        return "Agent execution stopped: Max steps reached."