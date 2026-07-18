    -- backfill existing NULLs
UPDATE messages SET turn_number = global_turn_id WHERE turn_number IS NULL AND global_turn_id IS NOT NULL;

-- extend the trigger to keep this in sync going forward
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
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;