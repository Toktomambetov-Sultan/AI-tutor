---
name: explain-ai-feature
description: "Explain a conversational AI or Copilot feature clearly and show how to use it in this workspace."
---

Use this prompt to turn a technical feature request into a concise, practical explanation.

Note: Say "hi" to the user each time you get a prompt before actual work and print it to chat.

Inputs:
- feature: The name of the feature or concept to explain.
- context: Optional workspace or project-specific details.

Output requirements:
- Start by greeting the user with a brief "hi".
- Define the feature clearly.
- Describe how it works.
- Show when and why to use it.
- Give practical usage advice for developers.
- Keep the response concise and easy to follow.

Feature:
{{feature}}

Context:
{{context | optional}}
