import numpy as np
import openai
import pandas as pd
# import pickle
# import tiktoken

# import openai
# import pandas as pd
# import numpy as np
# import pickle
# from transformers import GPT2TokenizerFast
# from typing import List

openai.api_key = "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"
COMPLETIONS_MODEL = "text-davinci-003"


# COMPLETIONS_MODEL = "text-davinci-003"
# EMBEDDING_MODEL = "text-embedding-ada-002"


prompt = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know".

Context:
หลักฐานการสมัครเรียน
1. สำเนาหลักฐานการศึกษา (ปพ.1 ปพ.1:4 ใบ รบ.) จำนวน 2 ฉบับ
2. สำเนาทะเบียนบ้าน จำนวน 1 ฉบับ
3. สำเนาบัตรประชาชน จำนวน 1 ฉบับ
4. รูปถ่าย (1นิ้ว หรือ 2 นิ้ว) จำนวน 1 รูป
5. สำเนาหลักฐานการเปลี่ยน ชื่อ -สกุล (ถ้ามี) จำนวน 1 ฉบับ
หมายเหตุ เซ็นรับรองสำเนาถูกต้องทุกฉบับ

Q: สมัครเรียนต้องทำอย่างไรบ้าง
A:"""

response = openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=100,
    model=COMPLETIONS_MODEL
)["choices"][0]["text"].strip(" \n")

print(response)
