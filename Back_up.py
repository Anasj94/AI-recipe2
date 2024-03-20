#Anas Javaid - 300299254
#Pratish Pushparaj - 300375330

import streamlit as st # Website
from transformers import pipeline, set_seed
from transformers import AutoTokenizer
from PIL import Image
import os
import time
import re
from utils import extention
import datetime
from openai import OpenAI #OpenAI
from pytube import Search # Youtube
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Check if the API key is set in the environment variables
api_key = os.environ.get('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=api_key)

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def pure_comma_separation(list_str, return_list=True):
    r = unique_list([item.strip() for item in list_str.lower().split(",") if item.strip()])
    if return_list:
        return r
    return ", ".join(r)


class TextGeneration:
    def __init__(self):
        self.debug = False
        self.tokenizer = None
        self.generator = None
        self.task = "text2text-generation"
        self.model_name_or_path = "flax-community/t5-recipe-generation"
        set_seed(42)

#Function to prettify the text of the Title,ingredients and direction
    def prettifying_text(self, text):
        recipeMapping = {"<sep>": "--", "<section>": "\n"}
        patternMappedRecipe = "|".join(map(re.escape, recipeMapping.keys()))

        text = re.sub(
            patternMappedRecipe,
            lambda m: recipeMapping[m.group()],
            re.sub("|".join(self.tokenizer.all_special_tokens), "", text)
        )

        dataFetched = {"title": "", "ingredients": [], "directions": []}
        for section in text.split("\n"):
            section = section.strip()
            if section.startswith("title:"):
                dataFetched["title"] = " ".join(
                    [w.strip().capitalize() for w in section.replace("title:", "").strip().split() if w.strip()]
                )
            elif section.startswith("ingredients:"):
                dataFetched["ingredients"] = [s.strip() for s in section.replace("ingredients:", "").split('--')]
            elif section.startswith("directions:"):
                dataFetched["directions"] = [s.strip() for s in section.replace("directions:", "").split('--')]
            else:
                pass

        return dataFetched

#Fetching the pipeline
    def fetch_PL(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.generator = pipeline(self.task, model=self.model_name_or_path, tokenizer=self.model_name_or_path)

        self.fetch_PL()

    def generate(self, items, generation_kwargs):
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
    generator = TextGeneration()
    generator.load()
    return generator  # Add this line to return the generator instance



cook = { "max_length": 512, "min_length": 64,"no_repeat_ngram_size": 3,"do_sample": True,"top_k": 60, "top_p": 0.95,"num_return_sequences": 1}

#Function to generate images using OpenAI
def generate_image(prompt):

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

#Function to suggest videos based on title
def search_youtube(query):
    search_results = Search(query)
    video_urls = [video.watch_url for video in search_results.results[:5]]  # Limiting to top 5 results
    return video_urls


def main():
    st.set_page_config(
        page_title="Culinary Innovations: AI-Driven Recipe Generation",
        page_icon="🍽️",
        layout="wide",
    )

    generator = load_text_generator()


    tab1 ,tab2= st.tabs(["Home","Contact Us"])
    with tab1:
        
        st.image(Image.open("asset/images/uOttawa.png"), width=300)
    

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
                    food_image = generated_recipe["image"]
                    # food_image = load_image_from_url(food_image, rgba_mode=True, default_image=generator.no_food)
                    # food_image = image_to_base64(food_image)

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
                                st.write(video_url) 

                    with tab2:
                        st.title("Contact Us")
                        st.write("For any inquiries or feedback, please feel free to contact us at:")
                        st.write("- Email: ajava059@uottawa.ca , ppush035@uottawa.ca")
                        st.write("- Phone: +1234567890")
                        st.write("- Address: University of Ottawa")


if __name__ == "__main__":
    main()

