import streamlit as st
from transformers import pipeline, set_seed
from transformers import AutoTokenizer
from PIL import Image
import os
import time
import re
from local_functions import extention  # Renamed ext to extention for clarity
import datetime
from openai import OpenAI
from pytube import Search
from dotenv import load_dotenv
import base64
import requests


# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is set in the environment variables
api_key = os.environ.get('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

community = "flax-community"
t5 = "t5-recipe-generation"

def unique_list(seq):
    """Return unique elements of a list while preserving the order."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def pure_comma_separation(list_str, return_list=True):
    """
    Convert a comma-separated string to a list of items.

    Args:
        list_str (str): The comma-separated string.
        return_list (bool): Whether to return a list (default is True).

    Returns:
        list_str (str): Comma-separated string if return_list is False,
                        otherwise returns a list of items.
    """
    r = unique_list([item.strip() for item in list_str.lower().split(",") if item.strip()])
    if return_list:
        return r
    return ", ".join(r)

class TextGeneration:
    """Class for generating text using transformer models."""
    def __init__(self):
        self.debug = False
        self.tokenizer = None
        self.generator = None
        self.task = "text2text-generation"
        self.model_name_or_path = community+"/"+t5
        set_seed(42)

    def prettifying_text(self, text):
        """
        Prettify the text by replacing special tokens.

        Args:
            text (str): The text to be prettified.

        Returns:
            dict: A dictionary containing the title, ingredients, and directions.
        """
        recipe_mapping = {"<sep>": "--", "<section>": "\n"}
        pattern_mapped_recipe = "|".join(map(re.escape, recipe_mapping.keys()))

        text = re.sub(
            pattern_mapped_recipe,
            lambda m: recipe_mapping[m.group()],
            re.sub("|".join(self.tokenizer.all_special_tokens), "", text)
        )

        data_fetched = {"title": "", "ingredients": [], "directions": []}
        for section in text.split("\n"):
            section = section.strip()
            if section.startswith("title:"):
                data_fetched["title"] = " ".join(
                    [w.strip().capitalize() for w in section.replace("title:", "").strip().split() if w.strip()]
                )
            elif section.startswith("ingredients:"):
                data_fetched["ingredients"] = [s.strip() for s in section.replace("ingredients:", "").split('--')]
            elif section.startswith("directions:"):
                data_fetched["directions"] = [s.strip() for s in section.replace("directions:", "").split('--')]
            else:
                pass

        return data_fetched

    def fetch_PL(self):
        """Fetch the pipeline for text generation."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.generator = pipeline(self.task, model=self.model_name_or_path, tokenizer=self.model_name_or_path)

    def load(self):
        """Load the text generation model."""
        self.fetch_PL()

    def generate(self, items, generation_kwargs):
        """
        Generate text using the loaded model.

        Args:
            items (str): The input text to generate from.
            generation_kwargs (dict): Keyword arguments for text generation.

        Returns:
            dict: A dictionary containing the generated recipe.
        """
        if not self.debug:
            generation_kwargs["num_return_sequences"] = 1
            generation_kwargs["return_tensors"] = True
            generation_kwargs["return_text"] = False

            generated_ids = self.generator(
                items,
                **generation_kwargs,
            )[0]["generated_token_ids"]
            recipe = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            recipe = self.prettifying_text(recipe)

            recipe["image"] = None

        return recipe

# Initialize a global variable to hold the TextGeneration object
text_generator = None

def load_text_generator():
    """
    Load the text generation model.

    Returns:
        TextGeneration: The text generation model instance.
    """
    global text_generator
    if text_generator is None:
        text_generator = TextGeneration()
        text_generator.load()
    return text_generator

# Default parameters for generating text
cook = { 
    "max_length": 512, 
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60, 
    "top_p": 0.95,
    "num_return_sequences": 1
}


def main():
    """Main function for running the Streamlit application."""
    st.set_page_config(
        page_title="Culinary Innovations: AI-Driven Recipe Generation",
        page_icon="🍽️",
        layout="wide",
    )

    generator = load_text_generator()
    st.header("Culinary Innovations: AI-Driven Recipe Generation 🍽️")
    st.divider()
    tab1, tab2 = st.columns([1, 1])  # Split the layout into two columns to display both tabs
    
    with tab1:
        st.image(Image.open("asset/images/MCR_Logo-1024x608.png"), width=400)

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg"])
        
        if uploaded_file is not None:
            if uploaded_file.type.startswith('image/'):
                # Send image data to Flask backend
                files = {'file': uploaded_file}
                response = requests.post('http://localhost:5000/get-ingredients', files=files)

                if response.status_code == 200:
                    result = response.json().get('ingredients', [])
                    st.write("Detected Ingredients:")
                    st.write(result)
                    items = st.text_area(
                        'Insert your food items here (separated by `,`): ',result  # Set the default value of the text area to the result
                    )
                else:
                    st.error("Failed to process the image.")
        else:
            items = st.text_area(
                'Insert your food items here (separated by `,`): ',
                ""
            )

        recipe_button = st.button('Generate Recipe')

        st.markdown(
            "<hr />",
            unsafe_allow_html=True
        )

        if recipe_button:
            entered_items = st.empty()
            entered_items.markdown("**Ingredients:** " + items)
        
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            with st.spinner("We are working on generating recipe for you MCR Lab"):
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                
                if not isinstance(items, str) or not len(items) > 1:
                    entered_items.markdown(
                        f"** Our cook would like to know your ingredients "
                    )
                else:
                    gen_kw = cook
                    generated_recipe = generator.generate(items, gen_kw)

                    title = generated_recipe["title"]

                    ingredients = extention.ingredients(
                        generated_recipe["ingredients"],
                        pure_comma_separation(items, return_list=True)
                    )
                    
                    # Generating image
                    response_generate_image = requests.post('http://localhost:5000/generate_image', data=title)
                    # Check if the request was successful (status code 200)
                    if response_generate_image.status_code == 200:
                        # Extract the list of video URLs from the response text
                        image_url = response_generate_image.json()['image_url']
                    else:
                        print("Error:", response_generate_image.text)
                    

                    # Search YouTube for related videos
                    query = title + " recipe"
                    response = requests.post('http://localhost:5000/search_youtube', data=query)

                    # Check if the request was successful (status code 200)
                    if response.status_code == 200:
                        # Extract the list of video URLs from the response text
                        video_urls = response.json()['video_urls']
                        print("Video URLs:", video_urls)
                    else:
                        print("Error:", response.text)


                    directions = extention.directions(generated_recipe["directions"])

                    with st.container():
                        # Information in the left column
                        st.markdown(
                            " ".join([
                                "<div class='r-text-recipe'>",
                                "<div class='food-title'>",
                                f"<img src='{image_url}' />",
                                f"<h2 class='font-title text-bold'>{title}</h2>",
                                "</div>",
                                '<div class="divider"><div class="divider-mask"></div></div>',
                                "<h3 class='ingredients font-body text-bold'>Ingredients</h3>",
                                "<ul class='ingredients-list font-body'>",
                                " ".join([f'<li>{item}</li>' for item in ingredients]),
                                "</ul>",
                                "</div>"
                            ]),
                            unsafe_allow_html=True
                        )

                        # Information in the right column
                        st.markdown(
                            " ".join([
                                "<h3 class='directions font-body text-bold'>How to Prepare</h3>",
                                "<ol class='ingredients-list font-body'>",
                                " ".join([f'<li>{item}</li>' for item in directions]),
                                "</ol>",
                                "</div>"
                            ]),
                            unsafe_allow_html=True
                        )

                        # Display search results
                        if "video_urls" in locals():
                            st.markdown("### Suggested Videos:")
                            for video_url in video_urls:
                                st.video(video_url)  # Display the video inline

    with tab2:  # Include the second tab for displaying contact information
        st.header("Information")
        st.write("Group 12 - Anas Javaid & Pratish Pushparaj")
        st.write("Submitting to Prof. Abdulmotaleb El Saddik (FRSC, FIEEE, FCAE, FEIC)")
        st.write("Multimedia Communication - ELG 5121")

if __name__ == "__main__":
    main()
