CREATE DATABASE IF NOT EXISTS ${OUTPUT_DATABASE} LOCATION 's3://${OUTPUT_BUCKET}/database/${OUTPUT_DATABASE}';

CREATE OR REPLACE TEMPORARY VIEW listenraw AS
SELECT
    character_id,
    track_id,
    real_play_time,
    yyyymmdd
FROM flo_log.listen_log
WHERE yyyymmdd >= ${ST}
AND yyyymmdd <= ${ET}
AND full_play_time > 0
;

CREATE OR REPLACE TEMPORARY VIEW metafilter AS
SELECT
    track_id,
    rep_track_id,
    pop_score,
    recency,
    instrumental,
    album_type,
    style_id_list,
    artist_id_list
FROM flo_reco.track_all
;

CREATE OR REPLACE TEMPORARY VIEW listenraw_ AS
SELECT
    t1.character_id,
    t2.rep_track_id as track_id,
    t1.real_play_time,
    t1.dt
FROM (
    SELECT
        character_id,
        cast(track_id as string) as track_id,
        real_play_time,
        yyyymmdd as dt
    FROM listenraw
) t1
JOIN (
    SELECT m1.track_id, m1.rep_track_id
    FROM (
        SELECT *
        FROM metafilter
        WHERE artist_id_list[0] NOT IN ('80088133')
    ) m1
    LEFT OUTER JOIN (
        SELECT DISTINCT track_id
        FROM metafilter m1
        LATERAL VIEW explode(m1.style_id_list) tmp as style_id
        WHERE (style_id LIKE '11%%')
        OR (style_id LIKE '15%%')
        OR (style_id LIKE '16%%')
        OR (style_id = '70103')
        OR (style_id = '70121')
    ) m2
    ON m1.track_id = m2.track_id
    WHERE m2.track_id is null
) t2
ON t1.track_id = t2.track_id
;

CREATE OR REPLACE TEMPORARY VIEW listenpoint AS
SELECT
    character_no,
    track_id,
    CASE
        WHEN real_play_time > 60 THEN 'Y'
        ELSE 'N'
    END as 1min_yn,
    real_play_time,
    dt
FROM (
    SELECT
        character_id as character_no,
        track_id,
        real_play_time,
        dt
    FROM listenraw_
    WHERE character_id is not null
    AND real_play_time > 0
)
GROUP BY character_no, track_id, real_play_time, dt
SORT BY dt
;

DROP TABLE IF EXISTS ${OUTPUT_DATABASE}.${OUTPUT_META_TABLE};
CREATE EXTERNAL TABLE IF NOT EXISTS ${OUTPUT_DATABASE}.${OUTPUT_META_TABLE} (
    track_id BIGINT,
    pop_score FLOAT,
    recency FLOAT,
    instrumental TINYINT,
    rep TINYINT,
    album_type_rl TINYINT,
    album_type_cp TINYINT,
    album_type_sl TINYINT,
    album_type_ep TINYINT,
    album_type_bs TINYINT,
    album_type_os TINYINT,
    album_type_cv TINYINT,
    album_type_mf TINYINT,
    album_type_lv TINYINT,
    album_type_etc TINYINT
)
COMMENT 'user2track meta info'
STORED AS ORC
LOCATION 's3://${OUTPUT_BUCKET}/database/${OUTPUT_DATABASE}/${OUTPUT_META_TABLE}'
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE ${OUTPUT_DATABASE}.${OUTPUT_META_TABLE} (
SELECT
    t1.track_id as track_id,
    t1.pop_score as pop_score,
    t1.recency as recency,
    t1.instrumental as instrumental,
    CASE
        WHEN t1.rep_track_id = l1.track_id THEN 1
        ELSE 0
    END as rep,
    CASE
        WHEN t1.album_type = 'RL' THEN 1
        ELSE 0
    END as album_type_rl,
    CASE
        WHEN t1.album_type = 'CP' THEN 1
        ELSE 0
    END as album_type_cp,
    CASE
        WHEN t1.album_type = 'SL' THEN 1
        ELSE 0
    END as album_type_sl,
    CASE
        WHEN t1.album_type = 'EP' THEN 1
        ELSE 0
    END as album_type_ep,
    CASE
        WHEN t1.album_type = 'BS' THEN 1
        ELSE 0
    END as album_type_bs,
    CASE
        WHEN t1.album_type = 'OS' THEN 1
        ELSE 0
    END as album_type_os,
    CASE
        WHEN t1.album_type = 'CV' THEN 1
        ELSE 0
    END as album_type_cv,
    CASE
        WHEN t1.album_type = 'MF' THEN 1
        ELSE 0
    END as album_type_mf,
    CASE
        WHEN t1.album_type = 'LV' THEN 1
        ELSE 0
    END as album_type_lv,
    CASE
        WHEN t1.album_type NOT IN ('RL','CP','SL','EP', 'BS', 'OS', 'CV', 'MF', 'LV') THEN 1
        ELSE 0
    END as album_type_etc
FROM metafilter t1
JOIN (
    SELECT DISTINCT track_id
    FROM listenpoint
) l1
ON t1.track_id = l1.track_id
);

CREATE OR REPLACE TEMPORARY VIEW preprocstep1 AS
SELECT
    character_no,
    track_id,
    1min_yn,
    dt
FROM listenpoint
ORDER BY dt DESC
;

DROP VIEW listenpoint;
DROP VIEW listenraw_;
DROP VIEW listenraw;
DROP VIEW metafilter;

CREATE OR REPLACE TEMPORARY VIEW preprocstep2 AS
SELECT
    character_no,
    dt,
    row_number() OVER (PARTITION BY character_no ORDER BY dt) as seq
FROM preprocstep1
GROUP BY character_no, dt
ORDER BY character_no, seq
;

CREATE OR REPLACE TEMPORARY VIEW preprocstep3 AS
SELECT
    character_no,
    dt,
    dt_,
    seq_diff,
    CASE
        WHEN seq_diff <= 0 THEN 'prev'
        ELSE 'next'
    END as mark
FROM (
    SELECT
        t1.character_no,
        t1.dt,
        t1_.dt as dt_,
        (t1_.seq - t1.seq) as seq_diff
    FROM preprocstep2 t1
    JOIN preprocstep2 t1_
    ON t1.character_no = t1_.character_no)
WHERE seq_diff > -5
AND seq_diff <= 2
;

CREATE OR REPLACE TEMPORARY VIEW preprocstep4 AS
SELECT
    t1.character_no,
    t3.mark,
    t1.1min_yn,
    t1.track_id,
    t3.dt
FROM preprocstep1 t1
JOIN preprocstep3 t3
ON t1.character_no = t3.character_no
AND t1.dt = t3.dt_
SORT BY t3.dt
;

DROP VIEW preprocstep1;
DROP VIEW preprocstep2;

-- INFER
DROP TABLE IF EXISTS ${OUTPUT_DATABASE}.${OUTPUT_INFER_TABLE};
CREATE EXTERNAL TABLE IF NOT EXISTS ${OUTPUT_DATABASE}.${OUTPUT_INFER_TABLE} (
    character_no STRING,
    x_play_tracks STRING,
    x_skip_tracks STRING
)
COMMENT 'user2track inference dataset (character_no: user_id, x_play_tracks: listened, x_skip_tracks: listened under 60 secs)'
STORED AS ORC
LOCATION 's3://${OUTPUT_BUCKET}/database/${OUTPUT_DATABASE}/${OUTPUT_INFER_TABLE}'
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE ${OUTPUT_DATABASE}.${OUTPUT_INFER_TABLE} (
SELECT
    t1.character_no,
    t1.x_play,
    t1.x_skip
FROM (
    SELECT
        *,
        concat_ws('|', slice(prev_play_t, 1, 100)) as x_play,
        concat_ws('|', slice(prev_skip_t, 1, 100)) as x_skip
    FROM (
        SELECT
            character_no,
            collect_list(
                CASE
                    WHEN 1min_yn = 'Y' THEN track_id
                    ELSE null
                END) as prev_play_t,
            collect_list(
                CASE
                    WHEN 1min_yn = 'N' THEN track_id
                    ELSE null
                END) as prev_skip_t,
            size(collect_set(
                CASE
                    WHEN 1min_yn = 'Y' THEN track_id
                    ELSE null
                END)) as unique_listens
        FROM (
            SELECT
                DISTINCT character_no,
                1min_yn,
                track_id,
                dt
            FROM preprocstep4
            DISTRIBUTE BY character_no
            SORT BY dt DESC)
    GROUP BY character_no)
    WHERE unique_listens >= ${UNIQ_LISTEN}
) t1
WHERE t1.x_play is not null
ORDER BY t1.character_no
);

-- GT
DROP TABLE IF EXISTS ${OUTPUT_DATABASE}.${OUTPUT_GT_TABLE};
CREATE EXTERNAL TABLE IF NOT EXISTS ${OUTPUT_DATABASE}.${OUTPUT_GT_TABLE} (
    character_no STRING,
    x_play_tracks STRING,
    y_dt_list STRING
)
COMMENT 'user2track inference dataset (character_no: user_id, y_play_tracks: G.T, y_dt: G.T DateList)'
STORED AS ORC
LOCATION 's3://${OUTPUT_BUCKET}/database/${OUTPUT_DATABASE}/${OUTPUT_GT_TABLE}'
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE ${OUTPUT_DATABASE}.${OUTPUT_GT_TABLE} (
SELECT
    character_no,
    concat_ws('|', collect_list(concat_ws(':', track_id, play_cnt))) as x_play_tracks,
    concat_ws('|', ref_dt_list) as y_dt_list
FROM (
    SELECT
        t1.character_no,
        t2.ref_dt_list,
        t1.track_id,
        count(*) as play_cnt
    FROM (
        SELECT
            character_no,
            1min_yn,
            track_id,
            seq_diff,
            dt
        FROM preprocstep4) t1
    JOIN (
        SELECT
            t2_0.character_no,
            t2_0.recent_dt,
            t2_1.ref_dt_list
        FROM (
            SELECT
                character_no,
                max(dt) as recent_dt
            FROM preprocstep3
            GROUP BY character_no) t2_0
        JOIN (
            SELECT
                character_no,
                dt,
                collect_set(dt_) as ref_dt_list
            FROM preprocstep3
            WHERE mark = 'prev'
            GROUP BY character_no, dt) t2_1
        ON (t2_0.character_no = t2_1.character_no)
        AND (t2_0.recent_dt = t2_1.dt)) t2
    ON t1.character_no = t2.character_no
    AND t1.dt = t2.recent_dt
    WHERE (t1.mark = 'prev')
    AND (t1.1min_yn = 'Y')
    GROUP BY t1.character_no, t2.ref_dt_list, t1.track_id
    DISTRIBUTE BY character_no
    SORT BY character_no, play_cnt DESC) t3
GROUP BY character_no, ref_dt_list
ORDER BY character_no
);

DROP VIEW preprocstep3;

-- TRAIN
DROP TABLE IF EXISTS ${OUTPUT_DATABASE}.${OUTPUT_TRAIN_TABLE};
CREATE EXTERNAL TABLE IF NOT EXISTS ${OUTPUT_DATABASE}.${OUTPUT_TRAIN_TABLE} (
    character_no STRING,
    x_play_tracks STRING,
    x_skip_tracks STRING,
    y_play_tracks STRING
)
COMMENT 'user2track train dataset (character_no: user_id, x_play_tracks: listened, x_skip_tracks: listened under 60 secs, y_play_tracks: might like)'
STORED AS ORC
LOCATION 's3://${OUTPUT_BUCKET}/database/${OUTPUT_DATABASE}/${OUTPUT_TRAIN_TABLE}'
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE ${OUTPUT_DATABASE}.${OUTPUT_TRAIN_TABLE}(
SELECT character_no, x_play, x_skip, y_play
FROM (
    SELECT
        t1.character_no,
        t1.x_play,
        t1.x_skip,
        t1.y_play,
        row_number() over (partition by t1.character_no order by t1.dt desc) as char_cnt
    FROM (
        SELECT
            *,
            concat_ws('|', slice(prev_play_t, 1, 100)) as x_play,
            concat_ws('|', slice(prev_skip_t, 1, 100)) as x_skip,
            concat_ws('|', slice(next_play_t, 1, 100)) as y_play
        FROM (
            SELECT
                t1_0.character_no,
                t1_0.prev_play_t,
                t1_0.prev_skip_t,
                t1_1.next_play_t,
                t1_0.unique_listens
            FROM (
                SELECT
                    character_no,
                    dt,
                    collect_list(CASE WHEN 1min_yn = 'Y' THEN track_id ELSE null END) as prev_play_t,
                    slice(collect_list(CASE WHEN 1min_yn = 'N' THEN track_id ELSE null END) as prev_skip_t,
                    SIZE(collect_set(CASE WHEN 1min_yn = 'Y' THEN track_id ELSE null END)) as unique_listens
                FROM (
                    SELECT *
                    FROM preprocstep4
                    WHERE mark = 'prev'
                    DISTRIBUTE BY character_no, dt)
                    GROUP BY character_no, dt) t1_0
                JOIN (
                    SELECT
                        character_no,
                        dt,
                        collect_list(CASE WHEN 1min_yn = 'Y' THEN track_id ELSE null END) as next_play_t
                    FROM (
                        SELECT *
                        FROM preprocstep4
                        WHERE mark = 'next'
                        DISTRIBUTE BY character_no, dt)
                    GROUP BY character_no, dt) t1_1
                ON (t1_0.character_no = t1_1.character_no) AND (t1_0.dt = t1_1.dt)
            WHERE unique_listens > ${UNIQ_LISTEN})
    ) t1
    WHERE t1.x_play is not null
    AND t1.y_play is not null)
WHERE char_cnt
LIMIT 6000000
);

DROP VIEW preprocstep4;