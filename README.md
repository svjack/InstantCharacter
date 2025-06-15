<div align="center">
<h1>InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework
 </h1>




[**Jiale Tao**](https://github.com/JialeTao)<sup>1</sup> · 
[**Yanbing Zhang**](https://github.com/Monalissaa)<sup>1</sup> · 
[**Qixun Wang**](https://github.com/wangqixun)<sup>12✝</sup> · 
[**Yiji Cheng**](https://www.linkedin.com/in/yiji-cheng-a8b922213/)<sup>1</sup> · 
[**Haofan Wang**](https://haofanwang.github.io/)<sup>2</sup> · 
[**Xu Bai**](https://huggingface.co/baymin0220)<sup>2</sup> · 
Zhengguang Zhou <sup>12</sup> · 
[**Ruihuang Li**](https://scholar.google.com/citations?user=8CfyOtQAAAAJ&hl=zh-CN) <sup>1</sup> · 
[**Linqing Wang**](https://scholar.google.com/citations?user=Hy12lcEAAAAJ&hl=en) <sup>12</sup> · Chunyu Wang <sup>1</sup> · 
Qin Lin <sup>1</sup> · 
Qinglin Lu <sup>1*</sup>


<sup>1</sup>Hunyuan, Tencent · <sup>2</sup>InstantX Team

<sup>✝</sup>tech lead · <sup>*</sup>corresponding authors

<a href='https://instantcharacter.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://xxxxx'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/spaces/InstantX/InstantCharacter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<!-- [![GitHub](https://img.shields.io/github/stars/Instant/InstantCharacter?style=social)](https://github.com/Tencent/InstantCharacter) -->


</div>


InstantCharacter is an innovative, tuning-free method designed to achieve character-preserving generation from a single image, supporting a variety of downstream tasks.


<img src='assets/1_lite.png'>


<!-- | reference | flux | + lora-ghibli | + lora-makoto |
|:-----:|:-----:|:-----:|:-----:|
<img src="assets/girl.jpg"  width=300>|<img src="assets/flux_instantcharacter.png" width=300>|<img src="assets/flux_instantcharacter_style_ghibli.png" width=300>|<img src="assets/flux_instantcharacter_style_Makoto.png" width=300>| -->

# Base

```python
# !pip install -U transformers accelerate diffusers huggingface_hub
# !pip install -U timm

import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from huggingface_hub import hf_hub_download
import os
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = "Cat_Boy_AMu_o_InstantCharacter_Images_Captioned"
os.makedirs(output_dir, exist_ok=True)

# Initialize pipeline
ip_adapter_path = hf_hub_download(repo_id="tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456

pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.init_adapter(
    image_encoder_path=image_encoder_path,
    image_encoder_2_path=image_encoder_2_path,
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
)

# Load reference image
ref_image_path = 'cat_boy.png'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# List of actions
actions = ['clean the table before the date',
 'plan the next date',
 'type a message during the date',
 'lock the door with a key',
 'store coffee beans in a jar',
 'draw with a pen',
 'drink water during the date',
 'record videos during the date',
 'store oil for a cooking date',
 'store books in a box',
 'make calls with a phone',
 'turn the TV on and off with a remote control',
 'share your own culture',
 'serve coffee during the date',
 'express appreciation for the date',
 'store water in a bucket',
 'serve water in a cup',
 'serve drinks on a tray',
 'maintain polite conversation',
 'shave with a razor',
 'check health status with a thermometer',
 'serve breakfast on a tray',
 'serve snacks on a tray',
 'drink soup with a spoon',
 'store trash in a trash can',
 "respect the date's customs",
 'carry a bouquet of flowers',
 'serve noodles in a bowl',
 'store medicine in case of an emergency',
 'serve rice on a plate',
 'lock the door before leaving for the date',
 'serve steak on a plate',
 'change channels with a remote control',
 'carry money in a wallet',
 'serve dessert on a plate',
 'serve juice in a cup',
 'surf the internet on a computer',
 'serve desserts on a tray during the date',
 'pick up food with chopsticks during the date',
 'ensure food safety',
 'store candy for the date',
 'dry hands before the date',
 'answer calls with headphones',
 'trim nails with nail clippers',
 'store gifts in a box',
 'carry items in a backpack',
 'store nuts for a picnic date',
 'serve soup during the date',
 'store flowers for the date',
 'serve wine during the date',
 'store stationery in a box',
 'store fruits for a picnic date',
 'send a thank-you message',
 'shade from the sun with an umbrella',
 'store tea leaves in a jar',
 'style hair with a hairdryer',
 'store vinegar in a bottle',
 'store cookies for the date',
 'serve pasta on a plate',
 'serve stir-fried vegetables in a wok',
 'check the time before the date',
 'cut meat on a cutting board',
 'avoid excessive drinking',
 'show interest in the date',
 'store trash in a bucket',
 'order drinks at a cafe',
 'record videos with a camera',
 'store candy in a bag',
 'spin dry clothes in a washing machine',
 'brush teeth with a toothbrush',
 'wash clothes before the date',
 'play background music',
 'bake bread in an oven',
 'wear glasses to look more intellectual',
 'exchange contact information',
 'store documents in a folder',
 'shave before the date',
 'serve tea during the meeting',
 'type on a computer',
 'style hair before the date',
 'store vinegar for a cooking date',
 'cut nails with nail clippers',
 'peel fruit for a picnic date',
 'prepare emergency items',
 'decorate with a watch',
 'serve tea in a pot',
 'prepare ingredients on a cutting board',
 'serve coffee in a cup',
 'store oil in a bottle',
 'mix beverages with a blender',
 'mix ingredients for a cooking date',
 'drink beverages through a straw',
 'deal with a reservation issue',
 "pay attention to the date's preferences",
 'send text messages with a phone',
 'serve stew in a pot',
 'store water for a picnic date',
 'serve salad in a bowl',
 'practice good table manners',
 'serve steak during the date',
 'dispose of trash in a trash can',
 'store medicine in a pill bottle',
 'store trash in a bag',
 'showcase your hobbies',
 'serve dessert during the date',
 'cut vegetables for a cooking date',
 'dispose of trash after the date',
 'serve rice during the date',
 'serve soup in a bowl',
 'protect eyes with glasses',
 'sweep the floor with a broom',
 'write in a notebook',
 'store soy sauce for a cooking date',
 'store flowers in a basket',
 'reflect on the date',
 'clean between teeth with dental floss',
 'store wine in a bottle',
 'store medicine in a medicine bottle',
 'store snacks for the date',
 'wipe hands with a napkin during the date',
 'store toys in a basket',
 'sweep dust before the date',
 'trim nails before the date',
 'cut vegetables on a cutting board',
 'serve pizza on a plate',
 'store snacks in a bag',
 'stew soup in a pot',
 'store clothes for the date',
 'ensure the date location is safe',
 'watch videos on a computer',
 'handle sudden rain',
 'peel with a knife',
 'store honey in a bottle',
 'carry books in a backpack',
 'mop the floor with a mop',
 'prepare ingredients for a cooking date',
 'serve cake during the date',
 'store body wash in a bottle',
 'store spices for a cooking date',
 'measure body temperature with a thermometer',
 'clean the floor with a vacuum cleaner',
 'serve water in a pot',
 'store honey in a jar',
 'clean the carpet with a vacuum cleaner',
 'adjust the music volume during the date',
 'serve pasta during the date',
 'write with a pen',
 'cook rice in a pot',
 'cut rope with scissors',
 'cut a cake during the date',
 'cut vegetables with a knife',
 'store shampoo in a bottle',
 'serve juice during the date',
 'bake vegetables for a cooking date',
 'adjust appearance in the restroom',
 'carry items in a shopping bag',
 'serve wine in a cup',
 'wipe face during the date',
 'bake meat in an oven',
 'serve stir-fried vegetables during the date',
 'store vegetables in a bag',
 'listen to podcasts with headphones',
 'mix ingredients with a blender',
 'organize photos from the date',
 'remove food debris with dental floss',
 'clean the floor with a mop',
 'see clearly with glasses',
 'store nuts in a bag',
 'store chocolate in a box',
 'check the temperature for an outdoor date',
 'store jam for a breakfast date',
 'store a gift for the date',
 'serve coffee in a pot',
 'cut paper with scissors',
 'demonstrate your sense of responsibility',
 'answer calls during the date',
 'cut a ribbon for a gift',
 'stir beverages with a spoon',
 'cut cake with a knife and fork',
 'cook a meal for the date',
 'carry a wallet for the date',
 'defrost food in a microwave',
 'take notes with a pen',
 'serve tea in a cup',
 'cut nails before the date',
 'serve soup on a plate',
 'surf the internet for date ideas',
 'cut fruit for a picnic date',
 'store spices in a jar',
 'serve cake on a plate',
 'look in a mirror',
 'send text messages during the date',
 'listen to podcasts about dating',
 'carry cash for the date',
 'show your thoughtfulness',
 'carry shopping items in a shopping bag',
 'pick up noodles with chopsticks',
 'serve water during the date',
 'groom in front of a mirror before the date',
 'store oil in a bucket',
 'groom in front of a mirror',
 'pick up food with chopsticks',
 'take notes in a notebook',
 'drink juice through a straw',
 'eat with chopsticks during the date',
 'open the door with a key',
 'serve rice in a bowl',
 'serve milk during the date',
 'make a call to confirm the date',
 'carry items in a wallet',
 'share personal stories',
 'organize documents in a folder',
 'adjust the volume with a remote control',
 'take photos with a phone',
 'bake cookies for the date',
 'open the door for the date',
 'serve soup in a pot',
 'shade from the sun during an outdoor date',
 'store medicine in a box',
 'serve tea during the date',
 'choose an appropriate gift',
 'choose a romantic location',
 'store water in a bottle',
 'serve salad on a plate',
 'fix hair before the date',
 'show your sense of humor',
 'eat steak with a knife and fork',
 "learn about the date's cultural background",
 'draw a picture for the date',
 'take photos with a camera',
 'stir-fry vegetables in a wok',
 'clean the car before picking up the date',
 'discuss future plans',
 'store vegetables for a cooking date',
 'store documents for a formal date',
 'serve noodles on a plate',
 'heat food in a microwave',
 'serve milk in a cup',
 'cut fabric with scissors',
 'store jam in a jar',
 'store cologne for the date',
 'write a love letter',
 'carry a shopping bag for the date',
 'clean teeth with a toothbrush',
 'pack clothes in a bag',
 'store vegetables in a basket',
 'clean the floor before the date',
 'draw in a notebook',
 'sweep dust with a broom',
 'serve salad during the date',
 'write down feelings about the date',
 'dry hair with a towel',
 'eat ice cream with a spoon',
 'serve rice in a pot',
 'serve porridge in a bowl',
 'comb hair with a comb',
 'carry a book for a thoughtful gift',
 'cut meat with a knife',
 'store soy sauce in a bottle',
 'take photos during the date',
 'eat steak during the date',
 'dry hair with a hairdryer',
 'drink soup during the date',
 'groom hair with a comb',
 'check the time with a watch',
 'remove lipstick stains after the date',
 'share your achievements',
 'store fruits in a basket',
 'wipe face with a towel',
 "learn the date's language",
 'serve stew during the date',
 'carry cards in a wallet',
 'bake vegetables in an oven',
 'listen to music during the date',
 'store honey for a sweet date',
 'wash clothes in a washing machine',
 'avoid cultural conflicts',
 'maintain personal hygiene',
 'bake bread for the date',
 'set up the date scene',
 'serve utensils on a tray',
 'drink water through a straw',
 'store paint in a bucket',
 'wipe hands with a towel',
 "listen to the date's thoughts",
 'prepare a surprise element',
 'store perfume for the date',
 'adjust lighting for ambiance',
 'trim beard with a razor',
 'beat eggs with a blender',
 'store tea leaves for the date',
 'store gifts for the date',
 'listen to music with headphones',
 'serve noodles in a pot',
 'cut fruit with a knife and fork',
 'shield from rain with an umbrella',
 'store clothes in a basket',
 'prepare a gift for the date',
 'adjust plans for unexpected changes',
 'trim beard before the date',
 'store cookies in a box',
 'eat with chopsticks',
 "handle the date's sudden mood change"]

# Process each action with tqdm progress bar
for action in tqdm(actions, desc="Generating cat boy images"):
    # Create prompt with prefix
    prompt = f"A cat boy {action}"
    
    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=ref_image,
        subject_scale=0.9,
        generator=torch.manual_seed(seed),
    ).images[0]
    
    # Create filename base (replace spaces and special characters)
    filename_base = prompt.replace(" ", "_").replace("/", "-").replace("'", "")[:100]
    
    # Save image
    image_path = os.path.join(output_dir, f"{filename_base}.png")
    image.save(image_path)
    
    # Save prompt to text file
    txt_path = os.path.join(output_dir, f"{filename_base}.txt")
    with open(txt_path, "w") as f:
        f.write(prompt)

print(f"All images and prompts saved to {output_dir}")

```

# ShenHe Style 
```python
### git clone https://huggingface.co/svjack/FLUX_Shenhe_Lora

import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from huggingface_hub import hf_hub_download
import os
from tqdm import tqdm

# Create output directory if it doesn't exist
output_dir = "Cat_Boy_AMu_o_InstantCharacter_Images_Captioned_ShenheStyle"
os.makedirs(output_dir, exist_ok=True)

# Initialize pipeline
ip_adapter_path = hf_hub_download(repo_id="tencent/InstantCharacter", filename="instantcharacter_ip-adapter.bin")
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456

pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.init_adapter(
    image_encoder_path=image_encoder_path,
    image_encoder_2_path=image_encoder_2_path,
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
)

# Load reference image
ref_image_path = 'cat_boy.png'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# Load style LoRA
lora_file_path = 'FLUX_Shenhe_Lora/tj_f1_shenhe_v1.safetensors'
trigger = 'tj_sthenhe, '

# List of actions
actions = ['clean the table before the date',
 'plan the next date',
 'type a message during the date',
 'lock the door with a key',
 'store coffee beans in a jar',
 'draw with a pen',
 'drink water during the date',
 'record videos during the date',
 'store oil for a cooking date',
 'store books in a box',
 'make calls with a phone',
 'turn the TV on and off with a remote control',
 'share your own culture',
 'serve coffee during the date',
 'express appreciation for the date',
 'store water in a bucket',
 'serve water in a cup',
 'serve drinks on a tray',
 'maintain polite conversation',
 'shave with a razor',
 'check health status with a thermometer',
 'serve breakfast on a tray',
 'serve snacks on a tray',
 'drink soup with a spoon',
 'store trash in a trash can',
 "respect the date's customs",
 'carry a bouquet of flowers',
 'serve noodles in a bowl',
 'store medicine in case of an emergency',
 'serve rice on a plate',
 'lock the door before leaving for the date',
 'serve steak on a plate',
 'change channels with a remote control',
 'carry money in a wallet',
 'serve dessert on a plate',
 'serve juice in a cup',
 'surf the internet on a computer',
 'serve desserts on a tray during the date',
 'pick up food with chopsticks during the date',
 'ensure food safety',
 'store candy for the date',
 'dry hands before the date',
 'answer calls with headphones',
 'trim nails with nail clippers',
 'store gifts in a box',
 'carry items in a backpack',
 'store nuts for a picnic date',
 'serve soup during the date',
 'store flowers for the date',
 'serve wine during the date',
 'store stationery in a box',
 'store fruits for a picnic date',
 'send a thank-you message',
 'shade from the sun with an umbrella',
 'store tea leaves in a jar',
 'style hair with a hairdryer',
 'store vinegar in a bottle',
 'store cookies for the date',
 'serve pasta on a plate',
 'serve stir-fried vegetables in a wok',
 'check the time before the date',
 'cut meat on a cutting board',
 'avoid excessive drinking',
 'show interest in the date',
 'store trash in a bucket',
 'order drinks at a cafe',
 'record videos with a camera',
 'store candy in a bag',
 'spin dry clothes in a washing machine',
 'brush teeth with a toothbrush',
 'wash clothes before the date',
 'play background music',
 'bake bread in an oven',
 'wear glasses to look more intellectual',
 'exchange contact information',
 'store documents in a folder',
 'shave before the date',
 'serve tea during the meeting',
 'type on a computer',
 'style hair before the date',
 'store vinegar for a cooking date',
 'cut nails with nail clippers',
 'peel fruit for a picnic date',
 'prepare emergency items',
 'decorate with a watch',
 'serve tea in a pot',
 'prepare ingredients on a cutting board',
 'serve coffee in a cup',
 'store oil in a bottle',
 'mix beverages with a blender',
 'mix ingredients for a cooking date',
 'drink beverages through a straw',
 'deal with a reservation issue',
 "pay attention to the date's preferences",
 'send text messages with a phone',
 'serve stew in a pot',
 'store water for a picnic date',
 'serve salad in a bowl',
 'practice good table manners',
 'serve steak during the date',
 'dispose of trash in a trash can',
 'store medicine in a pill bottle',
 'store trash in a bag',
 'showcase your hobbies',
 'serve dessert during the date',
 'cut vegetables for a cooking date',
 'dispose of trash after the date',
 'serve rice during the date',
 'serve soup in a bowl',
 'protect eyes with glasses',
 'sweep the floor with a broom',
 'write in a notebook',
 'store soy sauce for a cooking date',
 'store flowers in a basket',
 'reflect on the date',
 'clean between teeth with dental floss',
 'store wine in a bottle',
 'store medicine in a medicine bottle',
 'store snacks for the date',
 'wipe hands with a napkin during the date',
 'store toys in a basket',
 'sweep dust before the date',
 'trim nails before the date',
 'cut vegetables on a cutting board',
 'serve pizza on a plate',
 'store snacks in a bag',
 'stew soup in a pot',
 'store clothes for the date',
 'ensure the date location is safe',
 'watch videos on a computer',
 'handle sudden rain',
 'peel with a knife',
 'store honey in a bottle',
 'carry books in a backpack',
 'mop the floor with a mop',
 'prepare ingredients for a cooking date',
 'serve cake during the date',
 'store body wash in a bottle',
 'store spices for a cooking date',
 'measure body temperature with a thermometer',
 'clean the floor with a vacuum cleaner',
 'serve water in a pot',
 'store honey in a jar',
 'clean the carpet with a vacuum cleaner',
 'adjust the music volume during the date',
 'serve pasta during the date',
 'write with a pen',
 'cook rice in a pot',
 'cut rope with scissors',
 'cut a cake during the date',
 'cut vegetables with a knife',
 'store shampoo in a bottle',
 'serve juice during the date',
 'bake vegetables for a cooking date',
 'adjust appearance in the restroom',
 'carry items in a shopping bag',
 'serve wine in a cup',
 'wipe face during the date',
 'bake meat in an oven',
 'serve stir-fried vegetables during the date',
 'store vegetables in a bag',
 'listen to podcasts with headphones',
 'mix ingredients with a blender',
 'organize photos from the date',
 'remove food debris with dental floss',
 'clean the floor with a mop',
 'see clearly with glasses',
 'store nuts in a bag',
 'store chocolate in a box',
 'check the temperature for an outdoor date',
 'store jam for a breakfast date',
 'store a gift for the date',
 'serve coffee in a pot',
 'cut paper with scissors',
 'demonstrate your sense of responsibility',
 'answer calls during the date',
 'cut a ribbon for a gift',
 'stir beverages with a spoon',
 'cut cake with a knife and fork',
 'cook a meal for the date',
 'carry a wallet for the date',
 'defrost food in a microwave',
 'take notes with a pen',
 'serve tea in a cup',
 'cut nails before the date',
 'serve soup on a plate',
 'surf the internet for date ideas',
 'cut fruit for a picnic date',
 'store spices in a jar',
 'serve cake on a plate',
 'look in a mirror',
 'send text messages during the date',
 'listen to podcasts about dating',
 'carry cash for the date',
 'show your thoughtfulness',
 'carry shopping items in a shopping bag',
 'pick up noodles with chopsticks',
 'serve water during the date',
 'groom in front of a mirror before the date',
 'store oil in a bucket',
 'groom in front of a mirror',
 'pick up food with chopsticks',
 'take notes in a notebook',
 'drink juice through a straw',
 'eat with chopsticks during the date',
 'open the door with a key',
 'serve rice in a bowl',
 'serve milk during the date',
 'make a call to confirm the date',
 'carry items in a wallet',
 'share personal stories',
 'organize documents in a folder',
 'adjust the volume with a remote control',
 'take photos with a phone',
 'bake cookies for the date',
 'open the door for the date',
 'serve soup in a pot',
 'shade from the sun during an outdoor date',
 'store medicine in a box',
 'serve tea during the date',
 'choose an appropriate gift',
 'choose a romantic location',
 'store water in a bottle',
 'serve salad on a plate',
 'fix hair before the date',
 'show your sense of humor',
 'eat steak with a knife and fork',
 "learn about the date's cultural background",
 'draw a picture for the date',
 'take photos with a camera',
 'stir-fry vegetables in a wok',
 'clean the car before picking up the date',
 'discuss future plans',
 'store vegetables for a cooking date',
 'store documents for a formal date',
 'serve noodles on a plate',
 'heat food in a microwave',
 'serve milk in a cup',
 'cut fabric with scissors',
 'store jam in a jar',
 'store cologne for the date',
 'write a love letter',
 'carry a shopping bag for the date',
 'clean teeth with a toothbrush',
 'pack clothes in a bag',
 'store vegetables in a basket',
 'clean the floor before the date',
 'draw in a notebook',
 'sweep dust with a broom',
 'serve salad during the date',
 'write down feelings about the date',
 'dry hair with a towel',
 'eat ice cream with a spoon',
 'serve rice in a pot',
 'serve porridge in a bowl',
 'comb hair with a comb',
 'carry a book for a thoughtful gift',
 'cut meat with a knife',
 'store soy sauce in a bottle',
 'take photos during the date',
 'eat steak during the date',
 'dry hair with a hairdryer',
 'drink soup during the date',
 'groom hair with a comb',
 'check the time with a watch',
 'remove lipstick stains after the date',
 'share your achievements',
 'store fruits in a basket',
 'wipe face with a towel',
 "learn the date's language",
 'serve stew during the date',
 'carry cards in a wallet',
 'bake vegetables in an oven',
 'listen to music during the date',
 'store honey for a sweet date',
 'wash clothes in a washing machine',
 'avoid cultural conflicts',
 'maintain personal hygiene',
 'bake bread for the date',
 'set up the date scene',
 'serve utensils on a tray',
 'drink water through a straw',
 'store paint in a bucket',
 'wipe hands with a towel',
 "listen to the date's thoughts",
 'prepare a surprise element',
 'store perfume for the date',
 'adjust lighting for ambiance',
 'trim beard with a razor',
 'beat eggs with a blender',
 'store tea leaves for the date',
 'store gifts for the date',
 'listen to music with headphones',
 'serve noodles in a pot',
 'cut fruit with a knife and fork',
 'shield from rain with an umbrella',
 'store clothes in a basket',
 'prepare a gift for the date',
 'adjust plans for unexpected changes',
 'trim beard before the date',
 'store cookies in a box',
 'eat with chopsticks',
 "handle the date's sudden mood change"]

# Process each action with tqdm progress bar
for action in tqdm(actions, desc="Generating cat boy images with Shenhe style"):
    # Create prompt with prefix and trigger
    prompt = f"{trigger}A cat boy {action}"
    
    # Generate image with style LoRA
    image = pipe.with_style_lora(
        lora_file_path=lora_file_path,
        trigger=trigger,
        prompt=prompt, 
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=ref_image,
        subject_scale=0.9,
        generator=torch.manual_seed(seed),
    ).images[0]
    
    # Create filename base (replace spaces and special characters)
    filename_base = prompt.replace(" ", "_").replace("/", "-").replace("'", "")[:100]
    
    # Save image
    image_path = os.path.join(output_dir, f"{filename_base}.png")
    image.save(image_path)
    
    # Save prompt to text file
    txt_path = os.path.join(output_dir, f"{filename_base}.txt")
    with open(txt_path, "w") as f:
        f.write(prompt)

print(f"All images and prompts saved to {output_dir}")
```


## Release

- [2025/04/21] 🔥 Thanks to [jax-explorer](https://github.com/jax-explorer) for providing the [ComfyUI Warpper](https://instantcharacter.github.io/). 


- [2025/04/18] 🔥 We release the [demo](https://huggingface.co/spaces/InstantX/InstantCharacter) [checkpoints](https://huggingface.co/InstantX/InstantCharacter/) and [code](https://github.com/Tencent/InstantCharacter).
<!-- - [2025/04/02] 🔥 We release the [technical report](https://xxxxxxx/). -->
- [2025/04/02] 🔥 We launch the [project page](https://instantcharacter.github.io/).


## Download

You can directly download the model from [Huggingface](https://huggingface.co/InstantX/InstantCharacter).
```shell
huggingface-cli download --resume-download Tencent/InstantCharacter --local-dir checkpoints --local-dir-use-symlinks False
```

If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.
```shell
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Tencent/InstantCharacter --local-dir checkpoints --local-dir-use-symlinks False
```

Once you have prepared all models, the folder tree should be like:

```
  .
  ├── assets
  ├── checkpoints
  ├── models
  ├── infer_demo.py
  ├── pipeline.py
  └── README.md
```


## Usage


```python
# !pip install transformers accelerate diffusers huggingface_cli
import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline

# Step 1 Load base model and adapter
ip_adapter_path = 'checkpoints/instantcharacter_ip-adapter.bin'
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456
pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)

# Step 2 Load reference image
ref_image_path = 'assets/girl.jpg'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# Step 3 Inference without style
prompt = "A girl is playing a guitar in street"
image = pipe(
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter.png")
```


You can use style lora
<img src='assets/style.png'>

```shell
# download style lora
huggingface-cli download --resume-download InstantX/FLUX.1-dev-LoRA-Ghibli  --local-dir checkpoints/style_lora/ --local-dir-use-symlinks False
huggingface-cli download --resume-download InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai  --local-dir checkpoints/style_lora/ --local-dir-use-symlinks False
```

```python
# You can also use other style lora

# Step 3 Inference with style
lora_file_path = 'checkpoints/style_lora/ghibli_style.safetensors'
trigger = 'ghibli style'
prompt = "A girl is playing a guitar in street"
image = pipe.with_style_lora(
    lora_file_path=lora_file_path,
    trigger=trigger,
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter_style_ghibli.png")


# Step 3 Inference with style
lora_file_path = 'checkpoints/style_lora/Makoto_Shinkai_style.safetensors'
trigger = 'Makoto Shinkai style'
prompt = "A girl is playing a guitar in street"
image = pipe.with_style_lora(
    lora_file_path=lora_file_path,
    trigger=trigger,
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux_instantcharacter_style_Makoto.png")
```

## More case
Animal character are relatively unstable.
<img src='assets/more_case.png'>




<!-- ## Star History -->

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=instantX-research/InstantCharacter&type=Date)](https://star-history.com/#instantX-research/InstantCharacter&Date) -->




## Acknowledgment
 - Our work is sponsored by [HuggingFace](https://huggingface.co) and [fal.ai](https://fal.ai).

<div align="center">
  <img src='assets/thanks_hf_fal.jpg' style='width:300px;'>
</div>

 - Thanks to the model JY Duan.

<div align="center">
  <img src='assets/thanks_jyduan.jpg' style='width:300px;'>
</div>

<img src='assets/show.png'>


## Cite
If you find InstantCharacter useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{tao2025instantcharacter,
  title={InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework},
  author={Tao, Jiale and Zhang, Yanbing and Wang, Qixun and Cheng, Yiji and Wang, Haofan and Bai, Xu and Zhou, Zhengguang and Li, Ruihuang and Wang, Linqing and Wang, Chunyu and others},
  journal={arXiv preprint arXiv:2504.12395},
  year={2025}
}
```


