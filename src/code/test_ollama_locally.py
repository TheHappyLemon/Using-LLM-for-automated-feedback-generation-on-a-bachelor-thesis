import ollama

prompt = "Make a fictional story about cat named kitty. Two sentences only. Format the story as a JSON object"

response = ollama.generate(
    model='gemma3:1b',
    prompt=prompt,
    options={'temperature': 0},
    logprobs=True,
    top_logprobs=5,
    format='json'
)

for each in response.logprobs:
    token = each.get('token', '')
    logprob = each.get('logprob', None)
    print(token, logprob)
    alternates_logprob = each.get('top_logprobs', [])
    print(alternates_logprob)
    #print([it['token'] for it in alternates_logprob])

print()
print(response.response)