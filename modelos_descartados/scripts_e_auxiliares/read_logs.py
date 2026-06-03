# -*- coding: utf-8 -*-
import json

lines_to_read = 30
with open(r'C:\Users\pedro\.gemini\antigravity\brain\f443e9a0-d521-4082-97f3-1e952f8c2002\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines[-lines_to_read:]:
    try:
        data = json.loads(line)
        step = data.get('step_index')
        source = data.get('source')
        type_ = data.get('type')
        status = data.get('status')
        content = data.get('content')
        tool_calls = data.get('tool_calls')
        
        print(f"Step {step} | Source: {source} | Type: {type_} | Status: {status}")
        if content:
            print(f"  Content: {content[:200]}...")
        if tool_calls:
            print(f"  Tool Calls: {tool_calls}")
    except Exception as e:
        print(f"Error parsing line: {e}")
