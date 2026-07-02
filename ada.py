
import google.generativeai as genai
import os
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')
r = model.generate_content('Who is Sam Bankman-Fried? Answer in one word or name only.')
print(r.text)
