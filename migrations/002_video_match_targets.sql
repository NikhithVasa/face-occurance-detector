-- Store which target image/person produced each video match interval.

ALTER TABLE video_face_matches
  ADD COLUMN IF NOT EXISTS target_index integer,
  ADD COLUMN IF NOT EXISTS target_s3_key text;

CREATE INDEX IF NOT EXISTS idx_vfm_target_index ON video_face_matches (video_id, target_index);