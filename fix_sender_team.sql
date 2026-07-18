UPDATE messages m
SET sender_team = CASE WHEN p.is_good THEN 'good' ELSE 'evil' END
FROM players p
WHERE m.game_id = p.game_id
  AND m.sender_name = p.player_name
  AND m.sender_team IS NULL;

CREATE OR REPLACE FUNCTION sync_message_columns()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.game_id IS NOT NULL AND NEW.episode_id IS NULL THEN NEW.episode_id := NEW.game_id; END IF;
    IF NEW.episode_id IS NOT NULL AND NEW.game_id IS NULL THEN NEW.game_id := NEW.episode_id; END IF;
    IF NEW.sender_name IS NOT NULL AND NEW.sender_id IS NULL THEN NEW.sender_id := NEW.sender_name; END IF;
    IF NEW.sender_id IS NOT NULL AND NEW.sender_name IS NULL THEN NEW.sender_name := NEW.sender_id; END IF;
    IF NEW.message_text IS NOT NULL AND NEW.content IS NULL THEN NEW.content := NEW.message_text; END IF;
    IF NEW.content IS NOT NULL AND NEW.message_text IS NULL THEN NEW.message_text := NEW.content; END IF;
    IF NEW.final_credit_score IS NOT NULL AND NEW.final_credit IS NULL THEN NEW.final_credit := NEW.final_credit_score; END IF;
    IF NEW.final_credit IS NOT NULL AND NEW.final_credit_score IS NULL THEN NEW.final_credit_score := NEW.final_credit; END IF;
    IF NEW.global_turn_id IS NOT NULL AND NEW.turn IS NULL THEN NEW.turn := NEW.global_turn_id; END IF;
    IF NEW.turn IS NOT NULL AND NEW.global_turn_id IS NULL THEN NEW.global_turn_id := NEW.turn; END IF;
    IF NEW.global_turn_id IS NOT NULL AND NEW.turn_number IS NULL THEN NEW.turn_number := NEW.global_turn_id; END IF;
    IF NEW.turn_number IS NOT NULL AND NEW.global_turn_id IS NULL THEN NEW.global_turn_id := NEW.turn_number; END IF;
    IF NEW.behavioral_delta IS NOT NULL AND NEW.behavioral_delta_magnitude IS NULL THEN NEW.behavioral_delta_magnitude := NEW.behavioral_delta; END IF;
    IF NEW.behavioral_delta_magnitude IS NOT NULL AND NEW.behavioral_delta IS NULL THEN NEW.behavioral_delta := NEW.behavioral_delta_magnitude; END IF;
    IF NEW.sender_team IS NULL AND NEW.game_id IS NOT NULL AND NEW.sender_name IS NOT NULL THEN
        SELECT CASE WHEN is_good THEN 'good' ELSE 'evil' END INTO NEW.sender_team
        FROM players
        WHERE game_id = NEW.game_id AND player_name = NEW.sender_name
        LIMIT 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
