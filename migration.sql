ALTER TABLE messages ADD COLUMN IF NOT EXISTS game_id VARCHAR(100);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_name VARCHAR(100);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS message_text TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS final_credit_score FLOAT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS global_turn_id INTEGER;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS proposal_round INTEGER;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS semantic_score FLOAT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS behavioral_delta FLOAT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP DEFAULT NOW();

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
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sync_columns ON messages;
CREATE TRIGGER sync_columns
BEFORE INSERT OR UPDATE ON messages
FOR EACH ROW EXECUTE FUNCTION sync_message_columns();