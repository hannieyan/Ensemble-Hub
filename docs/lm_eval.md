




```yaml
task: arc_challenge_chat
doc_to_text: 'Given the following question and four candidate answers (A, B, C and D), choose the best answer. Specifically, reply with "The best answer is X."\nQuestion: {{question.strip()}}\nA. {{choices.text[0]}}\nB. {{choices.text[1]}}\nC. {{choices.text[2]}}{% if choices.text|length > 3 %}\nD. {{choices.text[3]}}{% endif %}\n'
```







