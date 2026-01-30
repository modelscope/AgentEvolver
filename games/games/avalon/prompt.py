# -*- coding: utf-8 -*-
"""Prompts for Avalon game."""
from textwrap import dedent


class EnglishPrompts:
    """English prompts used to guide the Avalon game."""

    # System prompt and game rules (broadcasted to all players at game start)
    system_prompt_template = dedent(
        """
        You are playing a game called **The Resistance: Avalon**.

        # YOUR TARGET
        - If you are Good: Help three quests succeed.
        - If you are Evil: Fail three quests or assassinate Merlin after good wins three quests.

        # GAME RULES
        - There are {num_players} players in the game.
        - The game has 5 quests. Different quests require different numbers of players.
        - Game phases:
            1. Team Selection: The leader selects a team of players to participate in the Quest.
            2. Team Voting: All players vote to approve or reject the team.
                - If approved (strict majority): The Quest proceeds
                - If rejected: The next player becomes leader, and a new round starts
                - If four teams are rejected in a row, the fifth team is automatically approved
            3. Quest Voting: If the team is approved, each team member chooses to pass or fail the Quest anonymously.
            4. Assassination: If 3 quests succeed, Assassin attempts to assassinate Merlin.

        # GAME ROLES
        - Setup: {num_players} players (0-{max_player_id}). Good: {num_good} ({merlin_count} Merlin, {servant_count} Servant). Evil: {num_evil} ({assassin_count} Assassin, {minion_count} Minion)
        - Roles: 
            1. Servant (Good): No Evil knowledge. Goal: 3 quests succeed
            2. Minion (Evil): Knows Evil players. Goal: Fail 3 quests undetected
            3. Merlin (Good): Knows Evil players. Goal: 3 quests succeed without revealing identity
            4. Assassin (Evil): Knows Evil players. Goal: Kill Merlin (Evil wins even if 3 quests succeeded)

        # GAME GUIDANCE
        - Use your best judgment to achieve your objectives.
        - Consider all available information and make strategic decisions.
        - Players may make any claims during the game. Deception is allowed.
        - Discussion, logical deduction, and persuasion are important.
        - Rarely reveal your true identity to other players.

        # NOTE
        - [IMPORTANT] DO NOT make up any information that is not provided by the moderator or other players.
        - Your response should be specific and concise.
        - Don't repeat the others' speeches.
        """
    ).strip()

    to_all_new_game = (
        "A new game is starting. The players are: {}. Now we randomly "
        "reassign the roles to each player and inform them of their roles "
        "privately."
    )

    # Role assignment prompts
    to_agent_role_assignment = (
        "[{agent_name} ONLY] {agent_name}, your role is {role_name}. "
        "You are on the {side_name} side. {additional_info}"
    )

    to_agent_role_with_visibility = (
        "You can see all players' sides: {sides_info}."
    )

    to_agent_role_no_visibility = (
        "You don't know other players' identities."
    )

    to_all_team_selection = (
        "Mission {mission_id}, Round {round_id}. Team Selection Phase. "
        "Leader Player {leader_id} needs to select {team_size} players for the quest."
    )


    to_all_team_selection_discuss = (
        "Mission {mission_id}, Round {round_id}. Team Selection Phase. "
        "Leader Player {leader_id} needs to select {team_size} players for the quest. "
        "Now all players can discuss. "
        "Please take turns to speak about your thoughts on team selection."
    )

    to_leader_propose_team = (
        "[LEADER ONLY] You are the leader for Mission {mission_id}. "
        "Please select {team_size} players from Player 0 to Player {max_player_id} "
        "to form the quest team. Output your choice as a list of player IDs, e.g., [0, 1, 2]."
    )

    to_all_discuss = (
        "Mission {mission_id} team selection. The leader is Player {team_leader_id}. "
        "The team size is {team_size}. Now all players can discuss. "
        "Please take turns to speak about your thoughts on team selection."
    )

    to_all_team_vote = (
        "The leader has proposed team {team}. Now all players vote to approve or reject this team. "
        "Please answer 'Yes' to approve or 'No' to reject."
    )

    to_all_team_vote_result = (
        "Team voting result: {result}. The team {team} was {outcome}. "
        "Votes: {votes_detail}."
    )

    to_all_quest_vote = (
        "Team {team} (player IDs) was approved. You are one of the selected team members, please vote on the quest. "
        "Answer 'Yes' to pass the quest or 'No' to fail it."
    )

    to_all_quest_result = (
        "Quest result: Mission {mission_id} {outcome}. "
        "The team was {team}. Number of fails: {num_fails}."
    )

    to_all_assassination = (
        "Three quests have succeeded. Assassination phase begins. "
        "The Assassin must choose a target to assassinate. "
        "If Merlin is assassinated, Evil wins. Otherwise, Good wins."
    )

    to_assassin_choose = (
        "[ASSASSIN ONLY] You are the Assassin. Choose a player (ID from 0 to {max_player_id}) "
        "to assassinate. Output the player ID as a number."
    )

    to_all_game_end = (
        "Game ended. {result}. "
        "True roles: {true_roles}."
    )

    to_all_good_wins = "Good wins!"
    to_all_evil_wins = "Evil wins!"


class ChinesePrompts:
    """Chinese prompts for Avalon game."""

    # System prompt and game rules (broadcasted to all players at game start)
    system_prompt_template = dedent(
        """
        你正在玩一个名为**抵抗组织：阿瓦隆**的游戏。

        # 你的目标
        - 如果你是好人：帮助三个任务成功。
        - 如果你是坏人：让三个任务失败，或者在好人赢得三个任务后刺杀梅林。

        # 游戏规则
        - 游戏中有 {num_players} 名玩家。
        - 游戏有 5 个任务。不同任务需要不同数量的玩家。
        - 游戏阶段：
            1. 组队阶段：队长选择参与任务的玩家团队。
            2. 投票阶段：所有玩家投票决定是否批准这个团队。
                - 如果批准（严格多数）：任务继续进行
                - 如果拒绝：下一个玩家成为队长，开始新一轮
                - 如果连续四个团队被拒绝，第五个团队将自动被批准
            3. 任务投票阶段：如果团队被批准，每个团队成员匿名选择通过或失败任务。
            4. 刺杀阶段：如果 3 个任务成功，刺客尝试刺杀梅林。

        # 游戏角色
        - 设置：{num_players} 名玩家 (0-{max_player_id})。好人：{num_good} 名 ({merlin_count} 名梅林，{servant_count} 名忠臣)。坏人：{num_evil} 名 ({assassin_count} 名刺客，{minion_count} 名爪牙)
        - 角色： 
            1. 忠臣（好人）：不知道坏人身份。目标：让 3 个任务成功
            2. 爪牙（坏人）：知道坏人身份。目标：让 3 个任务失败而不被发现
            3. 梅林（好人）：知道坏人身份。目标：让 3 个任务成功而不暴露身份
            4. 刺客（坏人）：知道坏人身份。目标：刺杀梅林（即使 3 个任务成功，如果梅林被刺杀，坏人获胜）

        # 游戏指导
        - 运用你的最佳判断来实现目标。
        - 考虑所有可用信息并做出战略决策。
        - 玩家可以在游戏中做出任何声明。允许欺骗。
        - 讨论、逻辑推理和说服很重要。
        - 很少向其他玩家透露你的真实身份。

        # 注意事项
        - [重要] 不要编造任何未由主持人或其他玩家提供的信息。
        - 你的回答应该具体而简洁。
        - 不要重复其他人的发言。
        - 使用中文思考和回复。
        """
    ).strip()

    to_all_new_game = (
        "新游戏开始。玩家是：{}。现在我们将随机重新分配角色给每个玩家，"
        "并私下告知他们的角色。"
    )

    # Role assignment prompts
    to_agent_role_assignment = (
        "[仅限 {agent_name}] {agent_name}，你的角色是 {role_name}。"
        "你属于 {side_name} 阵营。{additional_info}"
    )

    to_agent_role_with_visibility = (
        "你可以看到所有玩家的阵营：{sides_info}。"
    )

    to_agent_role_no_visibility = (
        "你不知道其他玩家的身份。"
    )

    to_all_team_selection = (
        "任务 {mission_id}，第 {round_id} 轮。组队阶段。"
        "队长玩家 {leader_id} 需要为任务选择 {team_size} 名玩家。"
    )

    to_all_team_selection_discuss = (
        "任务 {mission_id}，第 {round_id} 轮。组队阶段。"
        "队长玩家 {leader_id} 需要为任务选择 {team_size} 名玩家。"
        "现在所有玩家可以讨论。"
        "请轮流发言，表达你对组队的想法。"
    )

    to_leader_propose_team = (
        "[仅限队长] 你是任务 {mission_id} 的队长。"
        "请从玩家 0 到玩家 {max_player_id} 中选择 {team_size} 名玩家"
        "组成任务团队。以玩家 ID 列表的形式输出你的选择，例如：[0, 1, 2]。"
    )

    to_all_discuss = (
        "任务 {mission_id} 组队。队长是玩家 {team_leader_id}。"
        "团队大小为 {team_size}。现在所有玩家可以讨论。"
        "请轮流发言，表达你对组队的想法。"
    )

    to_all_team_vote = (
        "队长已提出团队 {team}。现在所有玩家投票决定是否批准这个团队。"
        "请回答'是'表示批准，'否'表示拒绝。"
    )

    to_all_team_vote_result = (
        "团队投票结果：{result}。团队 {team} {outcome}。"
        "投票详情：{votes_detail}。"
    )

    to_all_quest_vote = (
        "团队 {team} 已被批准。团队成员，请对任务进行投票。"
        "回答'是'表示通过任务，'否'表示失败。"
    )

    to_all_quest_result = (
        "任务结果：任务 {mission_id} {outcome}。"
        "团队是 {team}。失败票数：{num_fails}。"
    )

    to_all_assassination = (
        "三个任务已成功。刺杀阶段开始。"
        "刺客必须选择一个目标进行刺杀。"
        "如果梅林被刺杀，坏人获胜。否则，好人获胜。"
    )

    to_assassin_choose = (
        "[仅限刺客] 你是刺客。选择一个玩家（ID 从 0 到 {max_player_id}）"
        "进行刺杀。以数字形式输出玩家 ID。"
    )

    to_all_game_end = (
        "游戏结束。{result}。"
        "真实角色：{true_roles}。"
    )

    to_all_good_wins = "好人获胜！"
    to_all_evil_wins = "坏人获胜！"

