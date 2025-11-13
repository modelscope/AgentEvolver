from dataclasses import dataclass
from typing import List
import json


@dataclass
class EnvEntityOpt:
    """Define an operation that can be performed on an entity."""
    name: str
    description: str


def get_crud_opts() -> List[EnvEntityOpt]:
    """
    Returns a list of standard CRUD (Create, Read, Update, Delete) operations.

    Returns:
        List[EnvEntityOpt]: A list of CRUD operations, each with a name and description.
    """
    return [
        EnvEntityOpt("create", "Create a new instance of this entity."),
        EnvEntityOpt("read", "Retrieve one or more attribute values of this entity."),
        EnvEntityOpt("update", "Modify one or more attribute values of this entity."),
        EnvEntityOpt("delete", "Remove an instance of this entity.")  # ⭐ Defines the delete operation
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
        return mapping[int(self._relation_difficulty)]  # ⭐ Maps the difficulty level to a descriptive string


class EnvProfile:
    """User profile and task environment description generator."""
    def __init__(self, name: str, background: str, task: TaskPreference):
        self._name = name
        self._background = background
        self._entities: List[EnvEntity] = []
        self._task_preference = task
        
        self._rubrics=[]
    
    def reg_rubric(self, rubric: str):
        self._rubrics.append(rubric)

    def reg_entity(self, entity: EnvEntity):
        """
        Registers a single environment entity.

        Args:
            entity (EnvEntity): The environment entity to be registered.
        """
        self._entities.append(entity)  # ⭐ Adds the given entity to the internal list

    def reg_entities(self, entities: List[EnvEntity]):
        """
        Registers multiple environment entities at once.

        Args:
            entities (List[EnvEntity]): A list of environment entities to be registered.
        """
        self._entities.extend(entities)  # ⭐ Extends the internal list with the provided entities

    def get_instruction(self) -> str:
        """
        Generate a **pure environment description** in English.
        This description contains NO role-setting for the LLM,
        so it can be seamlessly inserted into a larger prompt
        without causing conflicts.

        Returns:
            str: The generated environment description.
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

        return "\n".join(inst_parts)  # ⭐ Joins all parts to form the final instruction string

    def get_task_preference_instruction(self) -> str:
        """
        Generates an instruction string based on the user's task preferences.

        Returns:
            str: A formatted string describing the user's task preferences.
        """
        inst_parts = []
        inst_parts.append(f"The task should involve the following characteristics:")
        inst_parts.append(f"- **Average number of entities involved**: {self._task_preference.num_entities}")
        inst_parts.append(f"- **Average number of operations involved**: {self._task_preference.num_opts}")  # ⭐ Adds the number of operations to the instruction
        inst_parts.append(f"- **Relation difficulty**: {self._task_preference.relation_difficulty}")
        inst_parts.append("")
        inst_parts.append(f"**Rubrics**:")
        inst_parts.extend(self._rubrics)
        inst_parts.append("You are required to follow these preferences strictly.")
        inst_parts.append("")

        return "\n".join(inst_parts)  # ⭐ Joins the parts into a single string and returns it

    def to_json(self) -> str:
        """
        Converts the UserProfile object into a JSON formatted string.

        Returns:
            str: A JSON string representation of the UserProfile.
        """
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
                for entity in self._entities  # ⭐ Iterates over each entity in the _entities list to build the 'entities' part of the JSON
            ],
            "task_preference": {
                "num_entities": self._task_preference.num_entities,
                "num_opts": self._task_preference.num_opts,
                "relation_difficulty": self._task_preference._relation_difficulty
            }
        }
        return json.dumps(data, indent=2)  # ⭐ Converts the dictionary to a JSON string with an indentation of 2 spaces

    def save_to_json(self, file_path: str):
        """
        Serialize the UserProfile instance into a JSON file.

        Args:
            file_path (str): The path where the JSON file will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())  # ⭐ Write the JSON representation of the profile to the file

    @classmethod
    def from_json(cls, json_str: str) -> 'EnvProfile':
        """
        Deserialize a JSON string into a UserProfile object.

        Args:
            json_str (str): The JSON string representing a UserProfile.

        Returns:
            UserProfile: An instance of UserProfile reconstructed from the JSON string.
        """
        data = json.loads(json_str)  # ⭐ Parse the JSON string into a Python dictionary
        
        # Create task preference
        task_pref = TaskPreference(
            num_entities=data["task_preference"]["num_entities"],
            num_opts=data["task_preference"]["num_opts"],
            relation_difficulty=data["task_preference"]["relation_difficulty"]
        )
        
        # Create user profile
        env_profile = cls(
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
        
        env_profile.reg_entities(entities)  # ⭐ Register the entities with the user profile
        return env_profile

    @classmethod
    def load_from_json(cls, file_path: str) -> 'EnvProfile':
        """
        Loads a UserProfile instance from a given JSON file.

        Args:
            file_path (str): The path to the JSON file containing the user profile data.

        Returns:
            UserProfile: An instance of UserProfile initialized with the data from the JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:  # ⭐ Opens the file and prepares to read its contents
            return cls.from_json(f.read())  # ⭐ Converts the JSON string into a UserProfile instance


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

    user = EnvProfile(
        name="Xiaoming",
        background="A music enthusiast who enjoys playing songs based on mood.",
        task=task_pref
    )

    user.reg_entities([song_entity, account_entity])

    print(user.get_instruction())