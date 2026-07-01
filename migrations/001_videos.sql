-- Video face-detection storage.
-- Additive migration; mirrors existing conventions (uuid PKs via
-- gen_random_uuid(), album_id/album_event_id FKs, *_status text columns,
-- is_deleted/created_at/updated_at, jsonb for structured blobs).

CREATE TABLE IF NOT EXISTS videos (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  album_id            uuid,
  album_event_id      uuid,
  customer_id         uuid,
  file_name           text,
  original_s3_key     text,
  storage_album_slug  text,
  storage_event_slug  text,
  duration_sec        double precision,
  model               text,
  detection_params    jsonb DEFAULT '{}'::jsonb,
  target_person_id    uuid,
  target_s3_keys      jsonb,
  runpod_endpoint_id  text,
  runpod_job_id       text,
  detection_status    text DEFAULT 'pending',   -- pending|processing|completed|failed
  detection_error     text,
  result_json         jsonb,
  match_count         integer DEFAULT 0,
  is_deleted          boolean DEFAULT false,
  deleted_at          timestamp without time zone,
  created_at          timestamp without time zone DEFAULT now(),
  updated_at          timestamp without time zone DEFAULT now(),
  completed_at        timestamp without time zone
);

CREATE INDEX IF NOT EXISTS idx_videos_album ON videos (album_id);
CREATE INDEX IF NOT EXISTS idx_videos_event ON videos (album_event_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos (detection_status);
CREATE INDEX IF NOT EXISTS idx_videos_job ON videos (runpod_job_id);

CREATE TABLE IF NOT EXISTS video_face_matches (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id        uuid NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
  album_id        uuid,
  album_event_id  uuid,
  person_id       uuid,
  start_sec       double precision,
  end_sec         double precision,
  start_time      text,
  end_time        text,
  max_similarity  double precision,
  avg_similarity  double precision,
  frames_matched  integer,
  verified        boolean,
  created_at      timestamp without time zone DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vfm_video ON video_face_matches (video_id);
CREATE INDEX IF NOT EXISTS idx_vfm_album ON video_face_matches (album_id);
CREATE INDEX IF NOT EXISTS idx_vfm_event ON video_face_matches (album_event_id);
