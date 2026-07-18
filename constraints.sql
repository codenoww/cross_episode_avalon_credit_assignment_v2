-- games: does the games table even exist with a game_id column? Check first.
-- If it exists, add unique constraint:
ALTER TABLE games ADD CONSTRAINT games_game_id_unique UNIQUE (game_id);

-- players
ALTER TABLE players ADD CONSTRAINT players_game_player_unique UNIQUE (game_id, player_name);

-- messages
ALTER TABLE messages ADD CONSTRAINT messages_game_turn_unique UNIQUE (game_id, global_turn_id);

-- votes
ALTER TABLE votes ADD CONSTRAINT votes_game_mission_proposal_voter_unique UNIQUE (game_id, mission_number, proposal_id, voter_name);