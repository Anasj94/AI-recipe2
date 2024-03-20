from flask import Flask, request, jsonify
from openai import OpenAI
from pytube import Search
from dotenv import load_dotenv
import os
import base64
import requests
from PIL import Image
from io import BytesIO


# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

@app.route('/search_youtube', methods=['POST'])
def search_youtube():
    """
    Search for videos on YouTube based on a query.

    Returns:
        list: A list of URLs of top 5 search results.
    """
    try:
        query = request.data.decode('utf-8').strip()
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Search for videos on YouTube using pytube
        search_results = Search(query).results[:5]
        video_urls = [video.watch_url for video in search_results]
        return jsonify({'video_urls': video_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        # Get prompt from request
        prompt = request.data.decode('utf-8')  # Decode the raw bytes to UTF-8 string
        
        # Call the generate_image function
        images = []
        response = client.images.generate(prompt=prompt,
                                          n=1,
                                          size='256x256',
                                          response_format='url')
        for image in response.data:
            images.append(image.url)
        
        # Return generated image URL in JSON format
        return jsonify({'image_url': images[0] if images else None})
    except Exception as e:
        # Handle any errors
        return jsonify({'error': str(e)}), 500


# Function to make request to OpenAI API
def encode_image(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    return encoded_string

@app.route('/get-ingredients', methods=['POST'])
def get_ingredients():
    uploaded_file = request.files['file']
    if uploaded_file:
        image_path = "temp_image.jpg"
        uploaded_file.save(image_path)
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "what are the ingredients in this picture? Give each ingredient with comma separated only"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            ingredients = response.json()['choices'][0]['message']['content']
            return jsonify({'ingredients': ingredients})
        else:
            return jsonify({'error': 'Failed to process the image.'}), 500



if __name__ == '__main__':
    app.run(debug=True)
