"""
识别模型异常输出的工具函数
- 非 ASCII 字符
- 重复字符
- 特殊错误标记符号
"""

import re
from functools import cache

# 各白名单类别对应正则片段
WHITE_LIST_REGEX_PARTS = {
    # 常见符号
    'common_symbols': '‘’“”–—…•™©®°±µ′″℉℃·×',
    # 中文标点
    'chinese_punct': '，。！？、；：“”‘’（）【】《》（）——……「」『』',
    # emoji 范围
    'emoji': (
        '\U0001F300-\U0001F5FF'
        '\U0001F600-\U0001F64F'
        '\U0001F680-\U0001F6FF'
        '\U0001F700-\U0001F77F'
        '\U0001F780-\U0001F7FF'
        '\U0001F800-\U0001F8FF'
        '\U0001F900-\U0001F9FF'
        '\U0001FA00-\U0001FA6F'
        '\U0001FA70-\U0001FAFF'
        '\u2702-\u27B0'
        '\u24C2-\U0001F251'
    ),
    # 中文字符
    'chinese': (
        '\u4E00-\u9FFF'
        '\u3400-\u4DBF'
        '\U00020000-\U0002A6DF'
        '\U0002A700-\U0002B73F'
        '\U0002B740-\U0002B81F'
        '\U0002B820-\U0002CEAF'
        '\uF900-\uFAFF'
        '\U0002F800-\U0002FA1F'
    ),
}


@cache
def build_pattern(white_list):
    """根据白名单类别构造正则"""
    allowed_parts = ['\x00-\x7F']  # 所有 ASCII
    for name in white_list:
        if name in WHITE_LIST_REGEX_PARTS:
            allowed_parts.append(WHITE_LIST_REGEX_PARTS[name])
    # 把允许的范围合并为一个字符类，并用反向类匹配“不被允许的字符”
    allowed_class = ''.join(allowed_parts)
    pattern = f'[^{allowed_class}]'  # 匹配 不允许 的字符
    return re.compile(pattern)

def has_non_ascii(text, white_list=('common_symbols', 'emoji', 'chinese', 'chinese_punct')):
    pattern = build_pattern(white_list)
    return bool(pattern.search(text))

def has_repeat(token, remember_n_words=5, patience_max=10):
    record_words = []
    patience = patience_max
    for char in token:
        if char not in record_words:
            record_words += [char]
            if len(record_words) > remember_n_words:
                record_words = record_words[1:]
            patience = patience_max
        else:
            patience -= 1
            if patience <= 0:
                return True
    return False

def repetition_penalty_reward_scalar(completion, detail=False):

    if detail:
        result = {
            'has_non_ascii': has_non_ascii(completion),
            'has_repeat': has_repeat(completion.split(), remember_n_words=5, patience_max=10),
            'has_repeat_x': has_repeat(completion, remember_n_words=4, patience_max=200),
            'has_wrong_sp_token': '<|im_start|>' in completion,
            # 'non_ascii': {ch for ch in completion if ord(ch) > 127}
        }
        if has_non_ascii(completion):
            for char in completion:
                if has_non_ascii(char):
                    print(f"---")
                    print(f"found non-ascii char: {char} ord={ord(char)}")

        return result

    if '<|im_start|>' in completion:
        return -1.0

    # if has_non_ascii(completion):
    #     return -1.0

    if has_repeat(completion.split(), remember_n_words=5, patience_max=10):
        return -1.0

    if has_repeat(completion, remember_n_words=4, patience_max=200):
        return -1.0

    return 0

def repetition_penalty_reward_scalar_debug(completion):
    for i in range(len(completion)):
        p = completion[:i]
        result = repetition_penalty_reward_scalar(p)
        if result != 0:
            return completion
    return ""

if __name__ == "__main__":
    # 测试示例
    # print(repetition_penalty_reward_scalar("Hello world!"))  # 0
    # print(repetition_penalty_reward_scalar("Hello world! 😄"))  # 0
    # print(repetition_penalty_reward_scalar("Hello world! Hello world!"))  # -1.0
    # print(repetition_penalty_reward_scalar("你好，世界！"))  # -1.0
    # print(repetition_penalty_reward_scalar("Hello <|im_start|> world!"))  # -1.0
    assert repetition_penalty_reward_scalar("""
        playlist_songs` API to get the list of songs in a playlist.

        Let's first call `show_playlist_songs` to get the list of songs for a playlist and then calculate the total duration.

        Code:
        ```python
        # Function to get song duration from Spotify API
        def get_song_duration(song_id, access_token):
            song_info = apis.spotify.show_song(song_id=song_id, access_token=access_token)
            return song_info.get('duration_ms', 0) // 1000  # Convert ms to seconds

        # Filter playlists and calculate total duration
        suitable_playlists = []
        for playlist in playlists:
            playlist_id = playlist['playlist_id']
            songs = apis.spotify.show_playlist_songs(playlist_id=playlist_id, access_token=spotify_access_token)
            total_duration = sum(get_song_duration(song['song_id'], spotify_access_token) for song in songs)

            if total_duration >= duration_mins * 60:  # Convert minutes to seconds
                suitable_playlists.append((playlist, total_duration))

        print(f"Suitable playlists: {len(suitable_playlists)}")
        ```

        Let's execute this code to find the suitable playlist.  🚀🚀 😄😄
    """) == 0

    assert repetition_penalty_reward_scalar("""
        Hello <|im_start|> world!
    """) == -1


    assert repetition_penalty_reward_scalar("""
        def has_non_ascii(text):
        non_ascii_but_normal = ['‘', '’', '“', '”', '–', '—', '…', '•', '™', '©', '®', '°', '±', 'µ', '°', '′', '″', '℉', '℃']
        for t in non_ascii_but_normal:
            text = text.replace(t, '')
        return not text.isascii()


        improve this function with option write_list, enabling it exclude

        1. non_ascii_but_normal
        2. emoji
        3. chinese
        4. chinese 标点
        5. other normal chars you can think of
    """) == 0


    assert repetition_penalty_reward_scalar("""
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    """) == -1


    assert repetition_penalty_reward_scalar("""
        fewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwe
    """) == -1

    assert repetition_penalty_reward_scalar("""
        wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd
    """) == -1

    assert repetition_penalty_reward_scalar("""
        1
        游戏科学在科隆游戏展上发布新作品《黑神话：钟馗》，视频中有哪些信息值得关注？
        世上何尝有鬼？妖魔皆从心生。 台下魑魅台上仙，好煞两副面！ 门内一滩子糊涂账，门外哪个喊青天？ 日月朝暮空空悬，凭谁掌那生死权。 不顺人情不合道，不争功名不趋炎。 提剑也，提剑也， 要把这清浊辨！ 由游戏科学开发的黑神话系列第二部作品《黑神话：钟馗》，今日正式公布首支 CG 先导预告片，并已在 2025 科隆游戏展的展前发布会同步亮相。 本作是以中国民间传说中的著名角色「钟馗」为主要创意来源的单机·动作·角色扮演游戏。因尚处早期开发阶段，暂无实机内容展示。

        5883万热度分享
        游戏科学在科隆游戏展上发布新作品《黑神话：钟馗》，视频中有哪些信息值得关注？
        2
        冯骥发声「《黑神话：悟空》DLC 确实是个不坏的选择，但此时此刻我们更想做一款新作」如何评价他的选择？
        《黑神话：悟空》发售后有相当长一段时间，我过得云里雾里。 一个心心念近二十年的事情，终于等到一个结果。而这个结果，超出最初的预期太多。 按理说，应该满地打滚，应该天天轻哼。 遗憾的是人类底层的预设不是这样，强烈的正面情绪持续时间好像都特别短，快乐总是一眨眼就过去。 那段时间我脑子里真正挥之不去的，主要是迷茫、虚无与惶恐（我知道这么说很矫情，别开枪）。可无论我怎么为自己「快乐不起来」感到羞愧，这些情绪依然不受控制地袭来，而且汹涌澎湃——尤其是被淹没在「DLC 到底做没做 DLC 都有谁啥时候发 DLC」的时候。 作为一个职业的成年人，我也很善于把这些负面隐藏起来，说服自己打起精神，老老实实开始做 DLC。 因为我很清楚，催 DLC 的朋友，毫无疑问都是热爱黑猴的人，是喜欢游科的人，是把我们一路抬上山的人。 于是，发售后的大半年，我确定了一些方向，写了一些设定，开了一些会，团队按照「先做 DLC」的计划，正经 RUN 了起来。 如此直到今年的某一天，杨奇上午给我留言，说「有重要的事儿想请教下」。 当天我恰好有事白天不在公司，就约晚上回来再聊。 回来后见到他，我问的第一句话是，「你是不是不想做 DLC，想做新的？」 释然的，欣然的，顺理成章的，我俩一拍即合。 然后，开始陆续说服其他同事。 再然后，就有了今天你们看到的《黑神话：钟馗》。 DLC 当然是个不坏的选择，但此时此刻，我们更想先做一款黑神话的新作—— 新的英雄，新的玩法，新的视觉，新的技术，新的故事。 放开手脚，大胆尝试，不拘定法，从零开始。 也许很多人认为，DLC 很稳健，DLC 很安全，DLC 很清晰，DLC 会快一点。 但我看到的很多二创作品，就已经比我们之前的 DLC 思路更加上天下地飞扬不羁 同各位一样，我无比喜爱西游记中那个妖魔神佛的世界，所以悟空的传说在未来会以更完整更扎实的方式，准备妥当后，再正式回来。 《岩田先生》一书中，任天堂的老社长说：「在既有的延长线上，是没有未来的。」 有未知，才有惊喜；有挑战，才有乐趣。 游戏科学会带着大家的爱与愿，继续认真交付每一份新的答卷。

        1793万热度分享
        冯骥发声「《黑神话：悟空》DLC 确实是个不坏的选择，但此时此刻我们更想做一款新作」如何评价他的选择？
        3
        如何评价 DeepSeek 于 2025 年 8 月 19 日更新的 V3.1 版本？
        目前只在官方微信群中通知，其他渠道尚未发布相关信息。

        610 万热度分享
        如何评价 DeepSeek 于 2025 年 8 月 19 日更新的 V3.1 版本？
        4
        新
        微软 Win11 最新 KB5063878 更新导致特定情况下 SSD 和 HDD 故障，如何解决？
        IT 之家 8 月 17 日消息，微软上周发布了一个非常重要的
    """) == 0


    print("All tests passed!")


