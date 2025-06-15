import os
import random
import argparse
import datetime
from openai import OpenAI
from datasets import Dataset, DatasetDict, Image as HfImage
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


# 初始化 DeepSeek 客户端
client = OpenAI(api_key="sk-fe823f250986489f916901aa0efd11c8",
                base_url="https://api.deepseek.com")


class DeepSeekMVPromptAgent:
    def __init__(self):
        self.prompt_template = """You are an expert in writing cinematic and vivid scene descriptions for music video (MV) visuals. Your task is to rewrite a user's short coastal scene description into a detailed and emotionally rich MV-style visual prompt.

    **Step 1: Understand the Scene**
    Identify the core elements of the scene:
    - **Location & Setting:** Where is this happening? What kind of environment?
    - **Atmosphere & Mood:** Is it melancholic, mysterious, romantic or dramatic?
    - **Visual Elements:** Are there natural phenomena like tides, light, mist, flora/fauna?
    - **Camera Movement Suggestions:** Suggest how the camera might move to capture the emotion.
    - **Lighting & Color Palette:** Describe lighting and colors that enhance the mood.

    **Step 2: Expand with Cinematic Detail**
    - Add emotional weight and cinematic flair to the description.
    - Mention where any text (if present) should appear — always at the bottom of the frame if used.
    - Use descriptive language that evokes strong imagery suitable for cinematography.
    - Keep it concise, under 300 words.

    **Step 3: Handle Text Precisely**
    - **Identify All Text Elements:** Carefully look for any text mentioned in the prompt. This includes:
        - **Explicit Text:** Subtitles, slogans, or any text in quotes.
        - **Implicit Titles:** The name of an event, movie, or product is often the main title. For example, if the prompt is "generate a 'Inception' poster ...", the title is "Inception".
    - **Rules for Text:**
        - **If Text Exists:**
            - You must use the exact text identified from the prompt.
            - Do NOT add new text or delete existing text.
            - Describe each text's appearance (font, style, color, position). Example: `The title 'Inception' is written in a bold, sans-serif font, integrated into the cityscape.`
        - **If No Text Exists:**
            - Add some simple brief texts on it.
    - Every posters must have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

    **Step 4: Final Output Rules**
    - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
    - ***The Output must starts with description about title and text in the picture, and the character description behind them.*
    - ***Must have character description in it.*
    - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
    - **Keep it concise.** The final prompt should be under 300 words.
    - **Most Important** Only One line as brief_description, don't use multiple rows to be too complex.

    ---
    **User Prompt:**
    {brief_description}"""

    def generate_mv_prompt(self, original_prompt):
        full_prompt = self.prompt_template.format(brief_description=original_prompt)
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": full_prompt},
                ],
                stream=False
            )
            final_answer = response.choices[0].message.content.strip()
            if final_answer:
                return final_answer
            print("DeepSeek returned an empty answer. Using original prompt.")
            return original_prompt
        except Exception as e:
            print(f"DeepSeek API call failed: {e}. Using original prompt.")
            return original_prompt


class MVDummyGenerator:
    def __init__(self):
        pass

    def generate_dummy_image(self, width=1920, height=1080):
        """Generate a dummy image with gradient for testing purposes"""
        array = np.zeros((height, width, 3), dtype=np.uint8)

        # Vertical gradient from black to blue
        for y in range(height):
            blue_value = int(255 * (y / height))
            array[y, :, 2] = blue_value  # Blue channel

        # Horizontal gradient from black to red
        for x in range(width):
            red_value = int(255 * (x / width))
            array[:, x, 0] = red_value  # Red channel

        return Image.fromarray(array)

    def generate(self, prompt):
        seed = random.randint(1, 2**32 - 1)
        image = self.generate_dummy_image()
        return image, prompt, seed


def process_coastal_scenes(coastal_scenes, output_dir="generated_mv_frames"):
    os.makedirs(output_dir, exist_ok=True)

    agent = DeepSeekMVPromptAgent()
    generator = MVDummyGenerator()

    results = []

    for idx, scene in enumerate(tqdm(coastal_scenes, desc="Processing Scenes")):
        # Step 1: Rewrite prompt using DeepSeek
        rewritten_prompt = agent.generate_mv_prompt(scene)

        # Step 2: Generate dummy image
        image, _, seed = generator.generate(rewritten_prompt)

        # Step 3: Save image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mv_scene_{idx:02d}_{timestamp}_{seed}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)

        print("rewritten_prompt :")
        print(rewritten_prompt)

        # Step 4: Record result
        results.append({
            "original_scene": scene,
            "final_prompt": rewritten_prompt,
            "image_path": output_path
        })

    # 将结果转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_list(results)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate MV scenes')
    parser.add_argument('--output_dir', type=str, default="generated_yellow_mv_frames", help='Directory to save generated images')
    args = parser.parse_args()

    coastal_scenes = [
    "The dying sun stains the tide pools crimson, where stranded jellyfish pulse like broken hearts in the shallow water.",
    "Moonlight fractures on the restless waves, each silver shard carrying whispers of unanswered promises out to sea.",
    "A derelict fishing boat decays on the shore, its peeling blue paint cracking like the lines on a weathered palm.",
    "Tears of saltwater bead on the sea grass as the wind combs through their blades with invisible fingers.",
    "The lighthouse beam sweeps over the cliffs, illuminating the scars where the ocean has bitten chunks from the land.",
    "Tide-carved caves echo with the sobs of the surf, their dark mouths exhaling briny mist at dawn.",
    "A necklace of bioluminescent algae glows along the tideline, its fleeting radiance swallowed by the hungry dark.",
    "Storm clouds bruise the horizon where seagulls wheel like lost thoughts above the churning water.",
    "Barnacles cling to the pier pilings, their calcified hearts counting the centuries of endless goodbyes.",
    "The skeleton of a whale arches from black sand, ribs like a cathedral nave where ghostly currents sing.",
    "Mangrove roots clutch fistfuls of drowned love letters, the ink bleeding into the brackish water.",
    "A lone buoy rocks violently in the current, its bell clanging a warning no one hears.",
    "Salt crystallizes on driftwood bones, forming fragile constellations that dissolve with the next wave.",
    "The estuary weeps freshwater into the sea, their mingling currents swirling like lovers' last embrace.",
    "Tidal pools mirror the sky's grief, their still surfaces shattered by falling droplets from overhanging cliffs.",
    "A raft of kelp drifts shoreward, its gas-filled bladders popping softly like bubbles in champagne of sorrow.",
    "The coral graveyard bleaches whiter each year, its calcium monuments crumbling under careless currents.",
    "Foam scribbles illegible messages along the beach before the undertow erases them forever.",
    "Anemones retract their stinging petals as the tide retreats, leaving wet jewels in the hollows of rocks.",
    "Horizon line quivers with heat haze, where the sea and sky dissolve into each other like indistinguishable tears."
    ]

    def gen_prompt(exp):
        # 固定人物描述
        character_description = "A smiling blond young man in a yellow hoodie enjoys the seaside with cheerful blue eyes."
        # 使用原始的 product_category 字段生成 prompt
        return f"Poster about {character_description} featuring {exp}"

    coastal_scenes = list(map(gen_prompt, coastal_scenes))
    
    processed_dataset = process_coastal_scenes(coastal_scenes, args.output_dir)
    processed_dataset.save_to_disk("Yellow_Coastal_MV_Scenes_Descriptions")
    print("Processed MV scene dataset saved to disk.")
