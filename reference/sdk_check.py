"""
Checks Ember's API with the official OpenAI and Anthropic Python SDKs.
Start a server first:  ember serve   then:  python reference/sdk_check.py [base_url]
"""
import sys

import anthropic
import openai

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"

oai = openai.OpenAI(base_url=BASE + "/v1", api_key="unused")
print("models:", [m.id for m in oai.models.list().data])

r = oai.chat.completions.create(model="ember", max_tokens=40, temperature=0,
                                messages=[{"role": "user", "content": "Name three primary colors."}])
print("openai:", repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage.prompt_tokens, r.usage.completion_tokens)

parts = []
for chunk in oai.chat.completions.create(model="ember", max_tokens=40, temperature=0, stream=True,
                                         messages=[{"role": "user", "content": "Count from 1 to 8."}]):
    if chunk.choices and chunk.choices[0].delta.content:
        parts.append(chunk.choices[0].delta.content)
print("openai stream:", len(parts), "chunks:", repr("".join(parts)))

r = oai.chat.completions.create(model="ember", max_tokens=60, temperature=0, stop=["4"],
                                messages=[{"role": "user", "content": "Count from 1 to 8, one number per line."}])
print("openai stop:", repr(r.choices[0].message.content), r.choices[0].finish_reason)

r = oai.chat.completions.create(model="ember", max_tokens=1500, temperature=0.6, top_p=0.95,
                                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                                messages=[{"role": "user", "content": "What is 17 * 23?"}])
msg = r.choices[0].message
print("openai thinking:", len(getattr(msg, "reasoning_content", "") or ""), "chars of reasoning; answer:", repr(msg.content[-80:]))

ant = anthropic.Anthropic(base_url=BASE, api_key="unused")
m = ant.messages.create(model="ember", max_tokens=40, system="Answer in Romanian.",
                        messages=[{"role": "user", "content": "What is the capital of Italy?"}])
print("anthropic:", repr(m.content[0].text), m.stop_reason, m.usage.input_tokens, m.usage.output_tokens)

with ant.messages.stream(model="ember", max_tokens=40, messages=[{"role": "user", "content": "Say hello in French."}]) as s:
    text = "".join(s.text_stream)
    final = s.get_final_message()
print("anthropic stream:", repr(text), final.stop_reason)

m = ant.messages.create(model="ember", max_tokens=60, stop_sequences=["3"],
                        messages=[{"role": "user", "content": "Count from 1 to 8."}])
print("anthropic stop:", repr(m.content[0].text), m.stop_reason, repr(m.stop_sequence))
print("all SDK checks passed")
