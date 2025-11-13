from dataclasses import dataclass
from typing import List
from datetime import date

from agentevolver.module.task_manager.env_profiles import EnvEntity, EnvEntityOpt, TaskPreference, EnvProfile, get_crud_opts


product_entity = EnvEntity(
    name="Product",
    description=(
        "An item available for purchase in the environment. "
        "Products may belong to categories such as:\n"
        "- **Electronics**: headphones, smartphones, cameras, smartwatches, cables.\n"
        "- **Home & Furniture**: rugs, chairs, beds, desks, lamps, storage baskets.\n"
        "- **Clothing & Accessories**: jeans, t-shirts, coats, shoes, slippers, dresses.\n"
        "- **Beauty & Personal Care**: shampoos, perfumes, deodorants, foundations, lip glosses.\n"
        "- **Food & Beverages**: snacks, cereals, sauces, coffee, tea, juices.\n"
        "- **Party & Decorations**: cupcake toppers, cake toppers, wall art, posters.\n"
        "- **Health & Oral Care**: toothpaste, toothbrushes, flossers, tongue scrapers.\n"
        "- **Hair Products**: hair extensions, wigs, storage cases, combs, brushes.\n"
        "- **Other**: candles, gift baskets, sports gear, travel accessories."
    ),
    attrs={
        "Name": "The name or title of the product.",
        "Category": "The category or type, e.g., rug, perfume, furniture.",
        "Color": "The color of the product.",
        "Size": "The size or dimensions of the product.",
        "Material": "The material composition, e.g., cotton, stainless steel.",
        "Flavor/Scent": "The flavor or scent if the product is consumable or fragrant.",
        "Features": "Notable characteristics such as easy to clean, wireless, gluten-free.",
        "Price": "The price of the product."
    },
    opts=[
        EnvEntityOpt("search", "Find products that match given attributes."),
        EnvEntityOpt("select", "Choose a specific product based on preferences."),
    ]
)

user_pref = TaskPreference(num_entities=2, num_opts=2, relation_difficulty=3)

env_profile = EnvProfile(
    name="Generic Shopper",
    background="A user searching for a wide variety of consumer products across categories like electronics, clothing, home goods, and food.",
    task=user_pref
)

env_profile.reg_entity(product_entity)
