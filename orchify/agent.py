from typing import List, Dict, Any, Optional, Union, Generator

from orchify.llm   import LLMInterface
from orchify.tool  import Tool
from orchify.event import AgentEvent, ToolEvent, RuntimeEvent, EVENT_TYPES
from orchify.utils import safecode
from orchify.env   import req_model

import json


class Agent:
    def __init__(
        self,
        name: str,
        llm: LLMInterface,
        system_prompt: str = 'You are a helpful assistant.',
        tools: Optional[List[Tool]] = None,
        model: str = req_model('gpt-4o'),
    ):
        self.name = name
        self.code = safecode(length=4)
        self.llm = llm
        self.system_prompt = system_prompt
        self.model = model
        
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        
        self.messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt}
        ]

    def run(
        self,
        user_input: str,
        max_steps: int = 5
    ) -> Generator[Union[AgentEvent, ToolEvent, RuntimeEvent], None, None]:
        turn_id = safecode(length=4)

        accumulated_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'prompt_cache_hit_tokens': 0,
            'prompt_cache_miss_tokens': 0,
            'total_tokens': 0
        }

        yield RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='run:start',
            payload={'user_input': user_input},
            turn=1
        )

        self.messages.append({'role': 'user', 'content': user_input})
        
        yield AgentEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='agent:start',
            content=user_input,
            turn=1
        )

        last_executed_step = 1
        
        for step in range(1, max_steps + 1):
            last_executed_step = step

            if step > 1:
                yield RuntimeEvent(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    event_type='run:next',
                    payload={'step': step},
                    turn=step
                )
            
            tools_payload = [t.info for t in self.tools.values()] if self.tools else None
            
            response_gen = self.llm.request(
                messages=self.messages,
                model=self.model,
                tools=tools_payload,
            )
            
            final_status = None
            for response in response_gen:
                if response.is_final:
                    final_status = response.final_status
                    yield AgentEvent.build_from_resp(
                        response,
                        agent_name=self.name,
                        agent_code=self.code,
                        turn_id=turn_id,
                        turn=step
                    )
                else:
                    chunk = response.current_chunk
                    if chunk:
                        if chunk.is_assembly_tool:
                            tool_events = ToolEvent.build_from_resp(
                                response,
                                agent_name=self.name,
                                agent_code=self.code,
                                turn_id=turn_id,
                                turn=step
                            )
                            for te in tool_events: yield te
                        else:
                            yield AgentEvent.build_from_resp(
                                response,
                                agent_name=self.name,
                                agent_code=self.code,
                                turn_id=turn_id,
                                turn=step
                            )
            
            if final_status is None: break
            
            accumulated_usage['completion_tokens'] += final_status.completion_tokens
            accumulated_usage['prompt_tokens'] += final_status.prompt_tokens
            accumulated_usage['prompt_cache_hit_tokens'] += final_status.prompt_cache_hit_tokens
            accumulated_usage['prompt_cache_miss_tokens'] += final_status.prompt_cache_miss_tokens
            accumulated_usage['total_tokens'] += final_status.total_tokens
            
            assistant_msg = {
                'role': 'assistant',
                'content': final_status.content or None
            }
            
            if final_status.tool_calls:
                assistant_msg['tool_calls'] = final_status.tool_calls
            self.messages.append(assistant_msg)
            
            if not final_status.tool_calls: break
                
            for tool_call in final_status.tool_calls:
                tool_id = tool_call.get('id') or ''
                tool_name = tool_call.get('function', {}).get('name') or ''
                args_str = tool_call.get('function', {}).get('arguments') or '{}'
                
                try:
                    args = json.loads(args_str) if args_str.strip() else {}
                except Exception:
                    args = {}
                
                yield ToolEvent.build_call_start(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    args=args,
                    turn=step
                )
                
                tool = self.tools.get(tool_name)
                if tool:
                    try:
                        if isinstance(args, dict):
                            result = tool.execute(**args)
                        else:
                            result = tool.execute()
                        
                        if not isinstance(result, str):
                            result = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result = f"Error executing tool '{tool_name}': {str(e)}"
                else:
                    result = f"Tool '{tool_name}' not found."
                
                yield ToolEvent.build_call_finish(
                    agent_name=self.name,
                    agent_code=self.code,
                    turn_id=turn_id,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    result=result,
                    turn=step
                )
                
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_id,
                    'name': tool_name,
                    'content': result
                })
        
        yield RuntimeEvent(
            agent_name=self.name,
            agent_code=self.code,
            turn_id=turn_id,
            event_type='run:finish',
            payload={
                'total_steps': last_executed_step,
                'usage': accumulated_usage
            },
            turn=last_executed_step
        )