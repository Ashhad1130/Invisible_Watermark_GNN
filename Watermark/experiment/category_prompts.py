"""Multi-category prompts for watermark experiments.

Categories: landscape, human_face, teddy, cat, dog, mountains, garden, scenery
"""
import random

CATEGORY_PROMPTS = {
    "landscape": [
        "A breathtaking panoramic view of snow-capped mountains at golden hour with a crystal clear alpine lake",
        "Misty mountain peaks emerging from clouds at dawn with pine forests covering the lower slopes",
        "A dramatic seascape with waves crashing against rocky cliffs at sunset with vibrant orange and purple sky",
        "Rolling sand dunes in the Sahara desert with beautiful patterns created by wind",
        "A cascading waterfall flowing into a turquoise pool surrounded by lush tropical vegetation",
        "Vast salt flats creating a perfect mirror reflection of clouds and mountains",
        "Bioluminescent bay at night with glowing blue water and a starry sky above",
        "A coral atoll seen from above with rings of turquoise and deep blue water",
        "A desert canyon at night with the Milky Way spanning the sky overhead",
        "Dramatic fjord landscape with steep cliffs and deep blue water under overcast skies",
    ],
    "human_face": [
        "A simple portrait of a young woman with natural lighting, plain white background, neutral expression",
        "A simple portrait of an elderly man with kind eyes, soft natural light, plain background",
        "A simple portrait of a young boy smiling, daylight, plain background, close up face",
        "A simple portrait of a middle-aged woman with brown hair, neutral expression, soft lighting",
        "A simple portrait of a man in his thirties, looking at the camera, plain light background",
        "A simple close-up portrait of a teenage girl with short hair, natural daylight",
        "A simple portrait of an old woman with white hair, gentle smile, soft background",
        "A simple portrait of a young man with dark skin, looking forward, minimal background",
        "A simple headshot of a child with curly hair, neutral background, natural light",
        "A simple portrait of a woman wearing glasses, looking straight ahead, clean background",
    ],
    "teddy": [
        "A classic brown teddy bear sitting on a white shelf, soft lighting, simple background",
        "A fluffy stuffed teddy bear with a red bow on a wooden floor",
        "A worn vintage teddy bear on a white blanket, warm golden light",
        "A large plush teddy bear sitting in an armchair by a window",
        "A small brown teddy bear with button eyes on a plain white background",
        "A cute teddy bear with a heart-shaped nose on a child's bed",
        "A soft cream-colored teddy bear sitting upright, simple studio lighting",
        "A well-loved teddy bear with a patched ear on a knitted blanket",
        "Two teddy bears sitting together on a wooden bench outdoors",
        "A tiny teddy bear inside a gift box with ribbon, white background",
    ],
    "cat": [
        "A fluffy orange tabby cat sitting on a windowsill in warm sunlight",
        "A sleek black cat with green eyes lying on a wooden floor",
        "A white Persian cat grooming itself on a sofa",
        "A grey cat with bright blue eyes looking directly at the camera",
        "A striped tabby kitten playing with a ball of yarn",
        "A ginger cat curled up asleep on a soft cushion",
        "A Siamese cat sitting alert on a garden wall in daylight",
        "A calico cat walking through a sunlit garden",
        "A long-haired Maine Coon cat sitting regally on a bookshelf",
        "A small black-and-white kitten with curious eyes on a white background",
    ],
    "dog": [
        "A golden retriever puppy sitting on green grass in bright sunlight",
        "A black Labrador dog running on a sandy beach",
        "A fluffy white Samoyed dog in the snow looking happy",
        "A beagle with long floppy ears sitting and looking at the camera",
        "A German Shepherd standing alert in a park on a sunny day",
        "A small brown dachshund puppy on a white background",
        "A Border Collie running through a field of wildflowers",
        "A friendly Corgi with a big smile sitting on a wooden porch",
        "A shaggy Old English Sheepdog on a green lawn",
        "A Dalmatian dog sitting on a pavement in warm afternoon light",
    ],
    "mountains": [
        "Snow-capped alpine peaks reflected in a still glacial lake at sunrise",
        "Rugged rocky mountain summits above the treeline under a deep blue sky",
        "A winding trail through a mountain pass with wildflowers on both sides",
        "Dramatic jagged peaks with a waterfall cascading down the cliff face",
        "A panoramic view of rolling mountain ranges fading into distant haze",
        "A lone pine tree on a rocky mountain ledge overlooking a valley",
        "Steep granite walls of a mountain canyon with a river below",
        "Mountain peaks glowing pink and orange in the alpenglow at dusk",
        "A high altitude plateau with yaks grazing against Himalayan peaks",
        "Fresh snow on dark pine trees with mountain summits in the background",
    ],
    "garden": [
        "An English cottage garden overflowing with roses and foxgloves in summer",
        "A formal French garden with geometric hedges, fountains and flower beds",
        "A Japanese zen garden with raked gravel, moss and pruned trees",
        "A wild meadow garden with poppies and cornflowers in warm afternoon light",
        "A kitchen garden with raised beds of vegetables and herbs in sunlight",
        "A Victorian walled garden with espalier fruit trees and climbing roses",
        "A tropical garden with palm trees, exotic flowers and a koi pond",
        "A spring garden with tulips and daffodils in rows of bright colour",
        "A peaceful garden path bordered by lavender and box hedging",
        "A rooftop garden in the city with planters of herbs and flowers",
    ],
    "scenery": [
        "A peaceful countryside lane lined with ancient oak trees in autumn",
        "A small village nestled in a valley with farmland and church steeple",
        "Rolling green hills with sheep grazing under a cloudy blue sky",
        "A wooden footbridge over a babbling stream through a meadow",
        "A coastal cliff top path with wildflowers and the sea below",
        "A quiet forest glade with soft light filtering through the canopy",
        "An old stone wall running across a green hillside in morning mist",
        "A lake at dusk with a rowing boat moored at a wooden jetty",
        "A winding country road through fields of golden wheat at harvest time",
        "A riverside scene with weeping willows trailing into the calm water",
    ],
}

ALL_CATEGORIES = list(CATEGORY_PROMPTS.keys())


def get_random_prompts(n, seed=42):
    """Return n prompts sampled randomly across all categories (reproducible)."""
    rng = random.Random(seed)
    all_prompts = [(cat, p) for cat, prompts in CATEGORY_PROMPTS.items() for p in prompts]
    selected = rng.sample(all_prompts, min(n, len(all_prompts)))
    # If n > pool size, cycle with different seeds
    while len(selected) < n:
        seed += 1
        extra = rng.sample(all_prompts, min(n - len(selected), len(all_prompts)))
        selected += extra
    return [{"prompt": p, "category": cat} for cat, p in selected[:n]]


def get_prompts_flat(n, seed=42):
    """Return n prompt strings sampled randomly across all categories."""
    return [item["prompt"] for item in get_random_prompts(n, seed=seed)]
