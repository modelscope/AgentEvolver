import re

def preserve_chinese(text):
    # 使用正则表达式匹配所有中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # 将匹配到的中文字符拼接成一个字符串
    return ''.join(chinese_chars)


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")


    def zipngram_chinese(text: str, ngram_size: int):
        import jieba
        text = preserve_chinese(text)
        seg_list = list(jieba.cut(text))
        # print(seg_list)
        return zip(*[seg_list[i:] for i in range(ngram_size)])


    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram_chinese(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            if total == 0:
                rewards.append(0.0)
                continue

            # total 是定值（句子分词长度）
            # 假设 全部不重复 total = len(ngrams)，scaling ~= 0, reward=0
            # 假设 全部重复 len(ngrams) = 1, scaling ~= 1, reward=-1
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


if __name__ == "__main__":
    for ngram_size in range(0, 100):
        reward_fn = get_repetition_penalty_reward(ngram_size=ngram_size, max_penalty=-1.0)
        completions = [
            [
                {
                    "content": "<|thought_start|>好的，我现在需要处理用户的问题：“那CPO板块未来会涨还是跌呢？”。首先，我需要回顾之前的对话历史，确保回答连贯。用户之前询问了CPO板块的当天表现，我提供了截止到2025年3月10日的收盘数据，并提到台积电计划量产CPO产品的积极信号。然后，用户问到担忧的具体内容，我解释了国际大厂发布CPO交换机可能影响利润的担忧，但也提到中际旭创等公司在积极应对，台积电也在推进技术。\n\n现在用户直接问CPO板块的未来走势。根据平台提供的供给知识，CPO板块受益于光通信技术和AI发展的双重驱动，台积电与英伟达、博通合作量产CPO产品，Marvell也在AI加速器上取得突破。这些都显示行业有技术进步和市场需求增长的潜力。同时，行业市盈率低于近10年80%的时间，估值偏低，具备性价比。虽然主力资金近期净流出，但重要股东持续增持，市场讨论热度提升，情绪趋于积极。\n\n用户的风险等级是高风险，理财认知高，但CPO属于科技赛道，用户当前科技类可能已经超配。需要结合这些信息，提供一个既符合用户特征，又基于现有数据的回复。需要强调行业潜力和当前估值优势，同时提示市场波动性，建议根据个人风险承受能力决策。确保不能出现重复引导，比如再次推荐基金或询问兴趣，因为这已经在前两轮对话中提到过。需要保持专业性和准确性，不编造信息，只基于提供的资料。确保不能出现重复引导，比如再次推荐基金或询问兴趣，因为这已经在前两轮对话中提到过。需要保持专业性和准确性，不编造信息，只基于提供的资料。确保不能出现重复引导，比如再次推荐基金或询问兴趣，因为这已经在前两轮对话中提到过。需要保持专业性和准确性，不编造信息，只基于提供的资料。<|thought_end|>您关注的CPO板块确实展现出了一定的投资潜力。从当前市场情况来看，CPO行业受益于光通信技术和AI发展的双重驱动，展现出良好的投资潜力。近期，台积电宣布与英伟达、博通合作，预计下半年量产CPO产品，反映行业技术进步和市场需求的增长。同时，行业市盈率低于近10年80%的时间，估值偏低，具备较高的性价比。\\n不过，作为科技赛道，CPO板块的波动性相对较高，短期内市场情绪和资金流向可能会影响板块表现。如果您是短期投资者，可能会经历较大的价格波动；如果是长期投资者，可以考虑分批布局，捕捉行业长期增长潜力。\\n您是想更倾向于短期把握CPO板块的波动机会，还是希望通过长期持有来捕捉行业成长？我可以根据您的偏好，为您推荐几只相关基金。",
                },
            ],
            [
                {
                    "content": "今天，一则消息在科技圈热传——据外媒报道，在近日向白宫提交的一项政策建议中，OpenAI对DeepSeek进行了十分露骨的攻击。一边称DeepSeek是“另一个华为”，是“国家补贴”“国家控制”的，会受中国指示“操纵其模型以造成损害”，一边强烈呼吁对该机构及类似机构开发的中国AI模型实施禁令。OpenAI的提案并非临时起意。今年1月，特朗普撤销了此前美国AI行政命令《安全、可靠和值得信赖的人工智能开发与使用》，随后签署了一项新的行政命令，宣布“美国的政策是维持并提升全球人工智能领域的主导地位”，并要求在180天内提交一份AI行动计划。无独有偶，美国另一家AI巨头Anthropic也强烈建议政府加强AI领域出口管制。它们显然都是瞄准了这一计划，想要以“技术话语权”换取“政策制定话语权”。事实上，以妖魔化外国竞争对手说事，博得政府支持最大化，是美国科技公司的惯用伎俩。但如此情绪激烈、赤裸裸地迎合美国“零和博弈”的政治逻辑，高调呼吁“抵制”“封禁”某家外国公司，多少还是过于难看。更可笑的是，这些话还出自自诩“开放”的OpenAI之口。OpenAI曾是开源的代名词，GPT-2的全面开放一度被视为行业标杆。但自GPT-3起，这家公司就选择筑起闭源高墙，更通过API订阅制赚得盆满钵满。本以为，DeepSeek掀起的AI行业普惠与效率革命，会让美国AI公司反思自己的路径选择。但如今来看，科技霸权主义对美国政治、美国科技企业的毒害远比我们想象的大得多。“层层封锁，封不住孤勇者。”不管出于什么心态，美国AI公司或许是忘了DeepSeek是怎么“逆袭”的了。近些年，美国政府为了压制中国人工智能的发展，动作不断，先是禁止AI芯片霸主英伟达向中国出口高端AI芯片，后又推动AI大模型霸主OpenAI停止向中国开放。这种情况下，即便对中国AI发展最为乐观的人也不敢想象——硬件性能无法做到最优，那就卷架构、卷工程，最终竟能“柳暗花明又一村”，让美国精心堆砌的人工智能高墙濒临坍塌。OpenAI将DeepSeek比作华为，怕是也忘了，几乎体验过美国所有打压、制裁手段的华为，如今已然是“轻舟已过万重山”。就如华为创始人任正非所说的，无论美国再怎么打压，华为也不恨美国。华为原来就像羊，慢慢吃草，变得很胖；美国是狼，狼一追羊就跑，减肥了就有战斗力。从更大范围看，于封锁中坚定自主创新、国产自研的心气与斗志，一直都是中国科技发展独特的“孤勇叙事”。至于迟迟丢不掉霸权逻辑者，在封闭、狭隘的创新环境里打转转，所谓的技术领先优势又能保持多久呢？最为讽刺的是，在提案中OpenAI还在唱“推广民主AI”“确保AI发展惠及所有人”的高调。试问，打压他国AI技术发展，到底“民主”在哪？“普惠”在哪？更何况，DeepSeek坚定走开源之路，极大促进了全球AI行业的合作交流，正让“AI人人可用”的理想不再遥不可及。“闭源”打压“开源”，还好意思抢占道德高地，不得不说，美国科技精英玩起政治双标那套来，也很溜。管中窥豹，可见一斑。马斯克曾言辞激烈地抨击OpenAI：“封闭”且“贪婪”。而这，形容的何尝不是当下的美国呢？"
                }
            ]
        ]
        rewards = reward_fn(completions)
        print(ngram_size, rewards)
