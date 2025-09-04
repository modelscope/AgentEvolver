from dataclasses import dataclass
from typing import List
import json


@dataclass
class EnvEntityOpt:
    """Define an operation that can be performed on an entity."""
    name: str
    description: str


def get_crud_opts() -> List[EnvEntityOpt]:
    """Return a standard set of CRUD operations."""
    return [
        EnvEntityOpt("create", "Create a new instance of this entity."),
        EnvEntityOpt("read", "Retrieve one or more attribute values of this entity."),
        EnvEntityOpt("update", "Modify one or more attribute values of this entity."),
        EnvEntityOpt("delete", "Remove an instance of this entity.")
    ]


@dataclass
class EnvEntity:
    """Information entity in the environment."""
    name: str
    description: str
    attrs: dict[str, str]
    opts: List[EnvEntityOpt]


class TaskPreference:
    """Describe the characteristics of the task to be generated."""
    def __init__(self, num_entities: int, num_opts: int, relation_difficulty: int):
        self._num_entities = num_entities
        self._num_opts = num_opts
        self._relation_difficulty = relation_difficulty
        assert 1 <= self._relation_difficulty <= 3

    @property
    def num_entities(self) -> int:
        return self._num_entities

    @property
    def num_opts(self) -> int:
        return self._num_opts

    @property
    def relation_difficulty(self) -> str:
        mapping = {
            1: (
                "Easy: Involves only one entity or one attribute. "
                "No cross-entity or cross-attribute dependencies. "
            ),
            2: (
                "Medium: Involves multiple entities or attributes, "
                "but operations are independent of each other. "
                "No prerequisite conditions or sequential dependencies."
            ),
            3: (
                "Hard: Involves multiple entities or attributes, "
                "and operations require prior condition checks or "
                "depend on the results of previous steps. "
                "Requires reasoning and decision-making."
            )
        }
        return mapping[int(self._relation_difficulty)]


class UserProfile:
    """User profile and task environment description generator."""
    def __init__(self, name: str, background: str, task: TaskPreference):
        self._name = name
        self._background = background
        self._entities: List[EnvEntity] = []
        self._task_preference = task

    def reg_entity(self, entity: EnvEntity):
        self._entities.append(entity)

    def reg_entities(self, entities: List[EnvEntity]):
        self._entities.extend(entities)

    def get_instruction(self) -> str:
        """
        Generate a **pure environment description** in English.
        This description contains NO role-setting for the LLM,
        so it can be seamlessly inserted into a larger prompt
        without causing conflicts.
        """
        inst_parts = []

        inst_parts.append("### Environment Overview")
        inst_parts.append(
            f"- **User Name**: {self._name}\n"
            f"- **User Background**: {self._background}"
        )

        inst_parts.append("\n### Entities in the Environment")
        for e in self._entities:
            inst_parts.append(f"#### Entity: {e.name}")
            inst_parts.append(f"- Description: {e.description}")
            inst_parts.append("- Attributes:")
            for attr_name, attr_desc in e.attrs.items():
                inst_parts.append(f"  - **{attr_name}**: {attr_desc}")
            inst_parts.append("- Available Operations:")
            for opt in e.opts:
                inst_parts.append(f"  - **{opt.name}**: {opt.description}")
            inst_parts.append("")  # blank line for readability

        inst_parts.append("### Task Preferences")
        inst_parts.append(f"The task should involve the following characteristics:")
        inst_parts.append(f"- **Average number of entities involved**: {self._task_preference.num_entities}")
        inst_parts.append(f"- **Average number of operations involved**: {self._task_preference.num_opts}")
        inst_parts.append(f"- **Relation difficulty**: {self._task_preference.relation_difficulty}")

        return "\n".join(inst_parts)
    
    def get_task_preference_instruction(self) -> str:
        inst_parts = []
        inst_parts.append(f"The task should involve the following characteristics:")
        inst_parts.append(f"- **Average number of entities involved**: {self._task_preference.num_entities}")
        inst_parts.append(f"- **Average number of operations involved**: {self._task_preference.num_opts}")
        inst_parts.append(f"- **Relation difficulty**: {self._task_preference.relation_difficulty}")

        return "\n".join(inst_parts)
    
    def to_json(self) -> str:
        """Convert UserProfile to JSON string."""
        data = {
            "name": self._name,
            "background": self._background,
            "entities": [
                {
                    "name": entity.name,
                    "description": entity.description,
                    "attrs": entity.attrs,
                    "opts": [{"name": opt.name, "description": opt.description} for opt in entity.opts]
                }
                for entity in self._entities
            ],
            "task_preference": {
                "num_entities": self._task_preference.num_entities,
                "num_opts": self._task_preference.num_opts,
                "relation_difficulty": self._task_preference._relation_difficulty
            }
        }
        return json.dumps(data, indent=2)
    
    def save_to_json(self, file_path: str):
        """Save UserProfile to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserProfile':
        """Create UserProfile from JSON string."""
        data = json.loads(json_str)
        
        # Create task preference
        task_pref = TaskPreference(
            num_entities=data["task_preference"]["num_entities"],
            num_opts=data["task_preference"]["num_opts"],
            relation_difficulty=data["task_preference"]["relation_difficulty"]
        )
        
        # Create user profile
        user_profile = cls(
            name=data["name"],
            background=data["background"],
            task=task_pref
        )
        
        # Add entities
        entities = []
        for entity_data in data["entities"]:
            opts = [EnvEntityOpt(opt["name"], opt["description"]) for opt in entity_data["opts"]]
            entity = EnvEntity(
                name=entity_data["name"],
                description=entity_data["description"],
                attrs=entity_data["attrs"],
                opts=opts
            )
            entities.append(entity)
        
        user_profile.reg_entities(entities)
        return user_profile
    
    @classmethod
    def load_from_json(cls, file_path: str) -> 'UserProfile':
        """Load UserProfile from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


# ===== Example usage =====
if __name__ == "__main__":
    song_entity = EnvEntity(
        name="Song",
        description="A track entry in the music collection.",
        attrs={
            "Title": "The name of the song.",
            "Rating": "The user's rating for the song."
        },
        opts=get_crud_opts() + [EnvEntityOpt("play", "Play this song.")]
    )

    account_entity = EnvEntity(
        name="Account",
        description="The user's personal account.",
        attrs={
            "Name": "The name of the account.",
            "Balance": "The current balance of the account."
        },
        opts=get_crud_opts()
    )

    task_pref = TaskPreference(num_entities=2, num_opts=2, relation_difficulty=3)

    user = UserProfile(
        name="Xiaoming",
        background="A music enthusiast who enjoys playing songs based on mood.",
        task=task_pref
    )

    user.reg_entities([song_entity, account_entity])

    print(user.get_instruction())