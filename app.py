import openai 
import os
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

openai_api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__, template_folder='templates')

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        model = "text-davinci"
        user = str(request.args.get('text'))

        if "exit" in user:
            return "Goodbye!"

        task_2_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
            ]
        )
        data = task_2_response.choices[0].message.content
        return data

if __name__ == '__main__':
    app.run(debug=True)
