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

# Step 1.1, To manually configure the CPU offload mode.
# You may selectively designate which layers to employ the offload hook based on the available VRAM capacity of your GPU.
# The following configuration can reach about 22GB of VRAM usage on NVIDIA L20 (Ada arch)

pipe.to("cpu")
pipe._exclude_from_cpu_offload.extend([
    # 'vae',
    'text_encoder',
    # 'text_encoder_2',
])
pipe._exclude_layer_from_cpu_offload.extend([
    "transformer.pos_embed",
    "transformer.time_text_embed",
    "transformer.context_embedder",
    "transformer.x_embedder",
    "transformer.transformer_blocks",
    # "transformer.single_transformer_blocks",
    "transformer.norm_out",
    "transformer.proj_out",
])
pipe.enable_sequential_cpu_offload()

pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
    device=torch.device('cuda')
)

# Step 1.2 Optional inference acceleration
# You can set the TORCHINDUCTOR_CACHE_DIR in production environment.

torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 1024
torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

for layer in pipe.transformer.attn_processors.values():
    layer = torch.compile(
        layer,
        fullgraph=True,
        dynamic=True,
        mode="max-autotune",
        backend='inductor'
    )
pipe.transformer.single_transformer_blocks.compile(
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.transformer.transformer_blocks.compile(
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.vae = torch.compile(
    pipe.vae,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)
pipe.text_encoder = torch.compile(
    pipe.text_encoder,
    fullgraph=True,
    dynamic=True,
    mode="max-autotune",
    backend='inductor'
)

import os
from itertools import product

# Create output directory
output_dir = "coastal_scene_outputs"
os.makedirs(output_dir, exist_ok=True)

# Reference images to cycle through
reference_images = ['苏锐.jpeg', '陈旭阳.jpeg']

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

# Perform Cartesian product of scenes and reference images
for scene_idx, (scene, ref_img) in enumerate(product(coastal_scenes, reference_images)):
    # Construct prompt with Makoto style prefix
    prompt = f"Makoto anime style, {scene}"
    
    # Load current reference image
    ref_image_path = ref_img
    ref_image = Image.open(ref_image_path).convert('RGB')
    
    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=ref_image,
        subject_scale=0.9,
        generator=torch.manual_seed(seed),
        height=720,
        width=1280,
    ).images[0]
    
    # Save with descriptive filename
    character_name = os.path.splitext(ref_img)[0]
    output_filename = f"{output_dir}/{character_name}_scene_{scene_idx:02d}.png"
    image.save(output_filename)
    
    print(f"Generated: {output_filename}")

print("All image generations completed!")
