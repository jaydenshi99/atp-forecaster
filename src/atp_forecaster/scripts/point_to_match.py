# contains script for converting point level prediction to match level prediction

import random

def point_to_match_mc(spw_a, rpw_a, trials: int = 1000, best_of: int = 3):
    """
    Convert point level prediction to match level prediction using Monte Carlo simulation.
    """
    winners = []  # 1 if id_a wins, 0 if id_b wins

    def play_game(server):
        """Simulate a single game with full deuce/advantage; returns game winner (1 if A, 0 if B)."""
        points_a, points_b = 0, 0
        while True:
            p_a = spw_a if server == 0 else rpw_a  # prob A wins the point (serve or return)
            if random.random() < p_a:
                points_a += 1
            else:
                points_b += 1
            if (points_a >= 4 or points_b >= 4) and abs(points_a - points_b) >= 2:
                return 1 if points_a > points_b else 0

    def play_tiebreak(start_server):
        """
        Simulate a tiebreak to 7 (win by 2). Serving order:
        - First point: start_server
        - Then two points each, alternating.
        Returns winner (1 if A, 0 if B) and the next server for the following set
        (receiver of the first tiebreak point).
        """
        pts_a, pts_b = 0, 0
        first_receiver = 1 - start_server
        points_played = 0

        while True:
            if points_played == 0:
                server = start_server
            else:
                block = (points_played - 1) // 2
                server = first_receiver if block % 2 == 0 else start_server

            p_a = spw_a if server == 0 else rpw_a
            if random.random() < p_a:
                pts_a += 1
            else:
                pts_b += 1

            points_played += 1

            if (pts_a >= 7 or pts_b >= 7) and abs(pts_a - pts_b) >= 2:
                winner = 1 if pts_a > pts_b else 0
                next_server = first_receiver  # receiver of first TB point serves next game/set
                return winner, next_server

    def play_set(start_server):
        """Simulate a set with tiebreak at 6-6; returns winner (1 if A, 0 if B) and next server."""
        games_a, games_b = 0, 0
        server = start_server
        while True:
            # Play regular games until 6-6
            if games_a == 6 and games_b == 6:
                tb_winner, next_server = play_tiebreak(server)
                return (1, next_server) if tb_winner == 1 else (0, next_server)

            game_winner = play_game(server)
            if game_winner == 1:
                games_a += 1
            else:
                games_b += 1

            server = 1 - server  # alternate server next game

            if (games_a >= 6 or games_b >= 6) and abs(games_a - games_b) >= 2:
                return (1, server) if games_a > games_b else (0, server)

    sets_to_win = best_of // 2 + 1

    for _ in range(trials):
        sets_a = 0
        sets_b = 0
        server = random.choice([0, 1])  # 0: A serves first, 1: B serves first

        while sets_a < sets_to_win and sets_b < sets_to_win:
            set_winner, server = play_set(server)
            if set_winner == 1:
                sets_a += 1
            else:
                sets_b += 1

        winners.append(1 if sets_a > sets_b else 0)

    return sum(winners) / len(winners), winners


def point_to_match_dp(spw_a, rpw_a, best_of: int = 3):
    """
    Exact match win probability for player A using dynamic programming (no Monte Carlo).

    Args:
        spw_a: Probability A wins a point on A's serve.
        rpw_a: Probability A wins a point on return (B's serve).
        best_of: Number of sets (3 or 5 typically).

    Returns:
        float: Probability A wins the match.
    """
    sets_to_win = best_of // 2 + 1

    from functools import lru_cache

    def game_prob(p_point):
        """
        Closed-form probability A wins a standard game (with deuce) given point-win prob p_point.
        Avoids deep recursion seen in deuce loops.
        """
        p = p_point
        q = 1 - p
        # Win before deuce: 4-0, 4-1, 4-2
        win_4_0 = p**4
        win_4_1 = 4 * p**4 * q
        win_4_2 = 10 * p**4 * q**2
        pre_deuce = win_4_0 + win_4_1 + win_4_2
        # Reach deuce (3-3), then win from deuce which is p^2 / (p^2 + q^2)
        reach_deuce = 20 * p**3 * q**3
        win_from_deuce = p**2 / (p**2 + q**2)
        return pre_deuce + reach_deuce * win_from_deuce

    def tiebreak_prob(start_server, max_pts=20):
        """
        Probability A wins the tiebreak to 7 (win by 2) with alternating serve blocks.
        Bounded iterative DP to avoid recursion and infinite state explosion.
        """
        dp = [[0.0 for _ in range(max_pts + 1)] for _ in range(max_pts + 1)]

        for total in range(max_pts * 2, -1, -1):
            for a in range(max(0, total - max_pts), min(max_pts, total) + 1):
                b = total - a
                if b < 0 or b > max_pts:
                    continue

                # terminal conditions
                if (a >= 7 or b >= 7) and abs(a - b) >= 2:
                    dp[a][b] = 1.0 if a > b else 0.0
                    continue
                if a == max_pts or b == max_pts:
                    # safety cap: if we hit the bound without decision, approximate by current lead
                    if a > b:
                        dp[a][b] = 1.0
                    elif b > a:
                        dp[a][b] = 0.0
                    else:
                        dp[a][b] = 0.5
                    continue

                points_played = a + b
                first_receiver = 1 - start_server
                if points_played == 0:
                    server = start_server
                else:
                    block = (points_played - 1) // 2
                    server = first_receiver if block % 2 == 0 else start_server

                p_a_point = spw_a if server == 0 else rpw_a
                dp[a][b] = p_a_point * dp[a + 1][b] + (1 - p_a_point) * dp[a][b + 1]

        return dp[0][0]

    gwp_serve = game_prob(spw_a)
    gwp_return = game_prob(rpw_a)

    def match_prob():
        """
        Iterative DP over match states to avoid recursion depth issues.
        State: (sets_a, sets_b, games_a, games_b, server)
        """
        cache = {}
        # enumerate possible ranges
        max_sets = sets_to_win
        max_games = 7  # up to 7 because 7-6 after tiebreak
        states = []
        for sa in range(max_sets + 1):
            for sb in range(max_sets + 1):
                for ga in range(max_games + 1):
                    for gb in range(max_games + 1):
                        for srv in (0, 1):
                            states.append((sa, sb, ga, gb, srv))

        # process in reverse order of progression
        states.sort(key=lambda s: (s[0] + s[1], s[2] + s[3]), reverse=True)

        def is_terminal(sa, sb):
            if sa >= sets_to_win:
                return True, 1.0
            if sb >= sets_to_win:
                return True, 0.0
            return False, None

        # precompute tiebreak probs for both starting servers
        tb_cache = {0: tiebreak_prob(0), 1: tiebreak_prob(1)}

        for state in states:
            sa, sb, ga, gb, srv = state
            terminal, val = is_terminal(sa, sb)
            if terminal:
                cache[state] = val
                continue

            # Set already decided at 7-x or x-7 (post-tiebreak cap)
            if ga == 7 or gb == 7:
                next_srv = srv
                if ga > gb:
                    cache[state] = cache[(sa + 1, sb, 0, 0, next_srv)]
                else:
                    cache[state] = cache[(sa, sb + 1, 0, 0, next_srv)]
                continue

            # Set decided without tiebreak
            if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
                next_srv = srv
                if ga > gb:
                    cache[state] = cache[(sa + 1, sb, 0, 0, next_srv)]
                else:
                    cache[state] = cache[(sa, sb + 1, 0, 0, next_srv)]
                continue

            # Tiebreak at 6-6
            if ga == 6 and gb == 6:
                tb_win = tb_cache[srv]
                next_srv = 1 - srv  # receiver of first TB point serves next set's first game
                win_val = cache[(sa + 1, sb, 0, 0, next_srv)]
                lose_val = cache[(sa, sb + 1, 0, 0, next_srv)]
                cache[state] = tb_win * win_val + (1 - tb_win) * lose_val
                continue

            # Play next game
            p_game = gwp_serve if srv == 0 else gwp_return
            next_srv = 1 - srv
            win_state = (sa, sb, ga + 1, gb, next_srv)
            lose_state = (sa, sb, ga, gb + 1, next_srv)
            cache[state] = p_game * cache[win_state] + (1 - p_game) * cache[lose_state]

        return cache[(0, 0, 0, 0, 0)]

    return match_prob()