from beyondagent.schema.trajectory import Reward, Trajectory
from typing import List, Dict
import uuid as uuid_gen


class ContextManagerBase:
    """
    1. prepare_next_llm_context
    2. check_context_token_num_safe
    3. prepare_world_interaction
    4. save_init_input
    5. save_llm_output
    6. save_env_output
    7. remove_last_context
    8. generate_log
    9. group_tokenize
    """

    def save_init_input(self, init_input_arr: List):
        raise NotImplementedError

    def prepare_next_llm_context(self, **kwargs) -> List:
        raise NotImplementedError

    def prepare_world_interaction(self, **kwargs) -> str:
        raise NotImplementedError

    def save_llm_output(self, llm_output, **kwargs):
        raise NotImplementedError

    def save_env_output(self, env_output, **kwargs):
        raise NotImplementedError

    def group_tokenize(self):
        raise NotImplementedError



class ExtendedMessage:

    def __init__(
            self,
            author,
            role="assistant",
            content="",
            token_arr=[],
            token_begin_index=-1,
            token_end_index=-1,
            clip=False,
            clip_token_limit=8192,
            tokenizer=None,
            token_generator="manual",
            build_from_uuid="",
            uuid=None,
        ):
        self.author = author
        self.role = role
        self.content = content
        self.token_arr = token_arr
        self.token_begin_index = token_begin_index
        self.token_end_index = token_end_index
        # use property to ensure content is safe before use
        self._content_for_future = ""
        self._info = ""
        self.clip = clip
        if uuid is None:
            self.uuid = uuid_gen.uuid4().hex
        else:
            self.uuid = uuid
        self.build_from_uuid = build_from_uuid

        if not clip:
            self.generate_content_for_future(tokenizer=None, clip=False)
        else:
            self.generate_content_for_future(tokenizer=tokenizer, clip=True, clip_token_limit=clip_token_limit)
        self.eos_token_id = tokenizer.eos_token_id
        if token_generator == 'auto':
            dummy_msg = [ {"role": "assistant",  "content": "dummy text"} ]
            self.token_arr, _ = self.get_inc_simple(
               text_frag_from=tokenizer.apply_chat_template(dummy_msg, tokenize=False),
               text_frag_to=tokenizer.apply_chat_template(dummy_msg +
                    [ {"role": self.role,  "content": self.content_for_future} ], tokenize=False),
               tokenizer=tokenizer
            )

    @property
    def content_for_future(self):
        if self._content_for_future == "":
            self._content_for_future = "(Empty Content)"
            # raise ValueError("content_for_future is not set, or previous llm output is empty!")
        return self._content_for_future


    @property
    def need_training(self):
        NEED_TRAIN_AUTHORS = ["llm"]
        NON_TRAIN_AUTHORS = ["env", "initialization", "user", "memory", "llm(do_not_train)"]
        assert (self.author in NEED_TRAIN_AUTHORS) or (self.author in NON_TRAIN_AUTHORS) or (self.author.endswith('(discard)')), f"author {self.author} is not identified"
        return (self.author in NEED_TRAIN_AUTHORS)


    def generate_content_for_future(self, tokenizer, clip, clip_token_limit=-1):
        _content: str = self.content
        if clip:
            assert clip_token_limit > 0, "clip_token_limit must be set when clip is True"
            n_token = len(tokenizer(_content, return_tensors="pt", padding=False)["input_ids"][0])
            if n_token > clip_token_limit:
                # 8000 > 4000
                n_char = len(_content)  # 10,000
                eps = 100   # token
                preserve_percent = (clip_token_limit - eps) / n_token  # 3900 / 8000
                n_char_to_preserve = int(n_char * preserve_percent)
                _content = _content[:n_char_to_preserve] + "... truncate ..."
        self._content_for_future = _content


    def get_loss_mask(self, blackout_token_combo):
        def blackout_specific_token_ids_first_encounter(mask, arr, token_ids):
            index = find_sublist_indices(arr, token_ids, reverse=False)
            if index >= 0:
                for i in range(index, index+len(token_ids)): mask[i] = 0
            return mask

        def blackout_everything_after_eos_but_keep_eos(mask, token_arr, eos_token_id):
            eos_position = token_arr.index(eos_token_id) if eos_token_id in token_arr else -1
            if eos_position != -1:
                for i in range(eos_position + 1, len(mask)):
                    mask[i] = 0
            return mask

        if self.need_training:
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(msg_token_mask, self.token_arr, blackout_token_combo)
            msg_token_mask = blackout_everything_after_eos_but_keep_eos(mask=msg_token_mask, token_arr=self.token_arr, eos_token_id=self.eos_token_id)
            return msg_token_mask
        else:
            msg_token_mask = [0] * len(self.token_arr)
            return msg_token_mask

    def get_inc_simple(self, text_frag_from, text_frag_to, tokenizer):
        """
        Get the incremental token array from text_frag_from to text_frag_to.
        """
        tokenizer_output = tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg


def find_sublist_indices(large_list, small_list, reverse=False):
    small_len = len(small_list)
    if reverse:
        for i in reversed(range(len(large_list) - small_len + 1)):
            if large_list[i: i+small_len] == small_list:
                return i
    for i in range(len(large_list) - small_len + 1):
        if large_list[i: i+small_len] == small_list:
            return i
    return -1


def replace_token_ids(place_holder, replace_with, begin, end):
    _begin_index = find_sublist_indices(place_holder, begin) + len(begin)
    _end_index = find_sublist_indices(place_holder, end, reverse=True)

    if replace_with[-len(end):] == end: # remove end token
        replace_with = replace_with[:-len(end)]
    if replace_with[:len(begin)] == begin: # remove begin token
        replace_with = replace_with[len(begin):]

    final = place_holder[:_begin_index] + replace_with + place_holder[_end_index:]
    return final
