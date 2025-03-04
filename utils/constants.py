DEFAULT_RAW_ENV_KWARGS = {
    "area": (64, 64),  # world have 64x64 elems, this is the grid
    "view": (9, 9),  # the player sees 9x9 elems from the grid. 9x9 is default
    "size": (
        64,
        64,
    ),  # resolution of the image for the obs, if make it bigger: see the same view elems in better resolution.
    "reward": True,
    "length": 10000,
    "seed": None,
}

crafter_actions_map = {
    "noop": {"index": 0, "requirement": "Do nothing"},
    "move_left": {"index": 1, "requirement": "Flat ground left to the agent."},
    "move_right": {"index": 2, "requirement": "Flat ground right to the agent."},
    "move_up": {"index": 3, "requirement": "Flat ground above the agent."},
    "move_down": {"index": 4, "requirement": "Flat ground below the agent."},
    "do": {
        "index": 5,
        "requirement": "Facing creature or material and have necessary tool.",
    },
    "sleep": {"index": 6, "requirement": "Energy level is below maximum."},
    "place_stone": {"index": 7, "requirement": "Stone in inventory."},
    "place_table": {"index": 8, "requirement": "Wood in inventory."},
    "place_furnace": {"index": 9, "requirement": "Stone in inventory."},
    "place_plant": {"index": 10, "requirement": "Sapling in inventory."},
    "make_wood_pickaxe": {
        "index": 11,
        "requirement": "Nearby table. Wood in inventory.",
    },
    "make_stone_pickaxe": {
        "index": 12,
        "requirement": "Nearby table. Wood, stone in inventory.",
    },
    "make_iron_pickaxe": {
        "index": 13,
        "requirement": "Nearby table, furnace. Wood, coal, iron in inventory.",
    },
    "make_wood_sword": {"index": 14, "requirement": "Nearby table. Wood in inventory."},
    "make_stone_sword": {
        "index": 15,
        "requirement": "Nearby table. Wood, stone in inventory.",
    },
    "make_iron_sword": {
        "index": 16,
        "requirement": "Nearby table, furnace. Wood, coal, iron in inventory.",
    },
}


crafter_world_info = """
General Crafter World Info:
- Zombies: 5 HP, deal 7 damage if the player is sleeping, 2 damage otherwise.
- Cows and Skeletons: 3 HP each.
- Drinking: Perform action 'do' in front of water to add 1 drink to inventory and drink.
- Maximum Stats: 9 for health, food, drink, and energy.
- Death Conditions: Health, food, or drink reaching 0, or stepping into lava.
- Reward System:
- +1 for unlocking a new achievement.
- -0.1 when health decreases.
- +0.1 when health increases.
- 0 for all other time steps.
The sum of rewards per episode can range from -0.9 (losing all health without any achievements) to 22 (unlocking all achievements and keeping or restoring all health until the time limit is reached). A score of 21.1 or higher means that all achievements have been unlocked.

Achievements:
- collect_coal
- collect_diamond
- collect_drink
- collect_iron
- collect_sapling
- collect_stone
- collect_wood
- defeat_skeleton
- defeat_zombie
- eat_cow
- eat_plant
- make_iron_pickaxe
- make_iron_sword
- make_stone_pickaxe
- make_stone_sword
- make_wood_pickaxe
- make_wood_sword
- place_furnace
- place_plant
- place_stone
- place_table
- wake_up

Termination Conditions:
- The episode ends if the agent's health reaches zero or upon reaching the time limit=10,000 steps.

Energy Management:
- Sleep to regain energy when low.
- Periodically, the player consumes food, energy and drink, reducing inventory counts.
- Drinking is replenished by performing 'do' in front of water.

Materials:
- water, grass, stone, path, sand, tree, lava, coal, iron, diamond, table, furnace.

Walkable Surfaces:
- grass, path, sand.

Items:
- health: max 9, initial 9
- food: max 9, initial 9
- drink: max 9, initial 9
- energy: max 9, initial 9
- sapling: max 9, initial 0
- wood: max 9, initial 0
- stone: max 9, initial 0
- coal: max 9, initial 0
- iron: max 9, initial 0
- diamond: max 9, initial 0
- wood_pickaxe: max 9, initial 0
- stone_pickaxe: max 9, initial 0
- iron_pickaxe: max 9, initial 0
- wood_sword: max 9, initial 0
- stone_sword: max 9, initial 0
- iron_sword: max 9, initial 0

Collection Rules(by action 'do' in front of them):
- tree → wood (no requirements)
- stone → stone (requires wood_pickaxe)
- coal → coal (requires wood_pickaxe)
- iron → iron (requires stone_pickaxe)
- diamond → diamond (requires iron_pickaxe)
- water → drink (no requirements)
- grass → sapling (10% chance, no requirements)

Placement Rules:
- stone: Uses 1 stone, can be placed on grass, sand, path, water, lava.
- table: Uses 2 wood, can be placed on grass, sand, path.
- furnace: Uses 4 stone, can be placed on grass, sand, path.
- plant: Uses 1 sapling, can be placed on grass.

Crafting Rules:
- wood_pickaxe: 1 wood, requires table nearby.
- stone_pickaxe: 1 wood + 1 stone, requires table nearby.
- iron_pickaxe: 1 wood + 1 coal + 1 iron, requires table and furnace nearby.
- wood_sword: 1 wood, requires table nearby.
- stone_sword: 1 wood + 1 stone, requires table nearby.
- iron_sword: 1 wood + 1 coal + 1 iron, requires table and furnace nearby.
"""
