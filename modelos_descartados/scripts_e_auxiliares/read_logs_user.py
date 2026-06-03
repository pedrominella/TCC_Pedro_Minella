# -*- coding: utf-8 -*-
import json

with open(r'C:\Users\pedro\.gemini\antigravity\brain\f443e9a0-d521-4082-97f3-1e952f8c2002\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    try:
        data = json.loads(line)
        step = data.get('step_index')
        source = data.get('source')
        type_ = data.get('type')
        if type_ == 'USER_INPUT':
            content = data.get('content')
            print(f"Step {step} | USER: {content[:200]}")
    except Exception as e:
        pass
