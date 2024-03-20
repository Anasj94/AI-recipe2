#Anas Javaid - 300299254
#Pratish Pushparaj - 300375330

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

# Load environment variables from .env file
load_dotenv()

# Check if the OpenAI API key is set in the environment variables
api_key = os.environ.get('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

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
        self.model_name_or_path = "flax-community/t5-recipe-generation"
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

@st.cache_data(hash_funcs={TextGeneration: lambda x: None})
def load_text_generator():
    """Load the text generation model and cache it."""
    generator = TextGeneration()
    generator.load()
    return generator

# Default parameters for generating text
cook = { 
    # The maximum length of the generated text. Text longer than this will be truncated.
    "max_length": 512, 
    
    # The minimum length of the generated text. Text shorter than this will be discarded.
    "min_length": 64,
    
    # Prevents repetition of n-grams within the generated text. If a sequence of `n` tokens has been generated before, it will not be repeated.
    "no_repeat_ngram_size": 3,
    
    # If set to `True`, sampling will be used during text generation. Sampling involves randomly selecting the next token based on its probability distribution.
    "do_sample": True,
    
    # Controls the number of tokens to consider during sampling. Only the `top_k` most likely tokens are considered at each step of generation.
    "top_k": 60, 
    
    # Also known as nucleus sampling, it controls the cumulative probability for sampling. Only tokens with a cumulative probability up to `top_p` are considered during sampling.
    "top_p": 0.95,
    
    # Specifies the number of sequences to generate. If set to more than 1, multiple sequences will be generated for the given input.
    "num_return_sequences": 1
}


def generate_image(prompt):
    """
    Generate image using OpenAI.

    Args:
        prompt (str): The prompt for image generation.

    Returns:
        dict: A dictionary containing the generated image URL.
    """
    try:
        images = []
        response = client.images.generate(prompt=prompt,
                                          n=1,
                                          size='256x256',
                                          response_format='url')
        for image in response.data:
            images.append(image.url)
            
        return {'created': datetime.datetime.now(), 'images': images}
    except Exception as e:
        print(e)

def search_youtube(query):
    """
    Search for videos on YouTube based on a query.

    Args:
        query (str): The search query.

    Returns:
        list: A list of URLs of top 5 search results.
    """
    search_results = Search(query)
    video_urls = [video.watch_url for video in search_results.results[:5]]  # Limiting to top 5 results
    return video_urls

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

        prompt = "Random"
        if prompt == "Random":
            prompt_box = ""

        items = st.text_area(
            'Insert your food items here (separated by `,`): ',
            pure_comma_separation(prompt_box, return_list=False),
        )
        items = pure_comma_separation(items, return_list=False)
        entered_items = st.empty()

        recipe_button = st.button('Generate Recipe')

        st.markdown(
            "<hr />",
            unsafe_allow_html=True
        )
        if recipe_button:
            entered_items.markdown("**Ingredients:** " + items)
        
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            with st.spinner("We are working on generating recipe for you @uOttawa"):
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
                    # OpenAI        
                    response = generate_image(title)
                    if response:
                        print("Response created at:", response['created'])
                        images = response['images']
                        for image in images:
                            print(image)
                            image_url=image
                    else:
                        print("Error occurred while generating image.")

                    # Youtube
                    query = title + " with ingredients " + items
                    video_urls = search_youtube(query)

                    directions = extention.directions(generated_recipe["directions"])

                    r1, _ = st.columns([6, 2])

                    with r1:
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

                        with st.container():
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
