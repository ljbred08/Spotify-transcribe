"""API tests via FastAPI TestClient (no network, no models)."""
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_index_serves_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "Refrain" in res.text


def test_tracks_empty_listing():
    res = client.get("/api/tracks")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_job_roundtrip_and_unknown_track_404():
    res = client.post("/api/jobs", json={"url": "https://example.invalid/x"})
    assert res.status_code == 200
    job = res.json()
    assert job["status"] == "queued"
    assert client.get(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get("/api/tracks/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.delete("/api/tracks/nope").status_code == 404


def test_export_missing_track_is_404():
    res = client.get("/api/tracks/x/export", params={"fmt": "xml"})
    assert res.status_code == 404  # unknown track wins over unknown format
    # format validation itself is covered in test_exporters.py


def test_upload_rejects_non_audio():
    res = client.post("/api/tracks/upload",
                      files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "unsupported audio type" in res.json()["detail"]


def test_upload_rejects_empty_file():
    res = client.post("/api/tracks/upload",
                      files={"file": ("song.mp3", b"", "audio/mpeg")})
    assert res.status_code == 400
    assert "empty file" in res.json()["detail"]


def test_upload_creates_track_and_queues_job(tmp_path, monkeypatch):
    import app.config as cfg
    from app.models import JobStatus
    from app.pipeline import pipeline, store

    monkeypatch.setattr(pipeline, "_run_transcribe_with_timeout",
                        lambda job, path, **kw: (_ for _ in ()).throw(
                            AssertionError("transcription must not run in tests")))
    payload = b"not really audio, but the endpoint only stores bytes"
    res = client.post("/api/tracks/upload",
                      files={"file": ("01 - Test Song.mp3", payload, "audio/mpeg")})
    assert res.status_code == 200
    data = res.json()
    job_id, track_id = data["job"]["id"], data["trackId"]

    track = store.get_track(track_id)
    assert track is not None
    # title derived from filename (leading track number stripped)
    assert track.title == "Test Song"
    assert track.audio_file is not None and track.audio_file.exists()
    assert track.audio_file.read_bytes() == payload
    assert track.url == ""  # uploaded files have no source URL

    job = store.get_job(job_id)
    assert job is not None
    assert job.kind == "upload"
    assert job.track_id == track_id

    import time
    for _ in range(20):
        if store.get_job(job_id).status in (JobStatus.ERROR, JobStatus.DONE):
            break
        time.sleep(0.1)
    assert store.get_job(job_id).track_id == track_id

    # cleanup mirrors the DELETE endpoint: files + index entry
    from app.pipeline import delete_track_files
    audio_path = track.audio_file
    delete_track_files(track_id)
    store.delete_track(track_id)
    assert store.get_track(track_id) is None
    assert not audio_path.exists()
    assert cfg.DATA_DIR in audio_path.resolve().parents


def test_settings_roundtrip_show_timestamps():
    from app.settings import settings

    before = settings.get()["show_timestamps"]
    try:
        out = settings.update(show_timestamps=True)
        assert out["show_timestamps"] is True
        out = settings.update(show_timestamps=False)
        assert out["show_timestamps"] is False
    finally:
        settings.update(show_timestamps=before)


# --- inline editing (title/artist rename + lyric line edit) ----------------


def _make_editable_track():
    from app.models import Line, Track, Word
    from app.pipeline import store

    track = Track(id="edittest1", url="", title="Edit Me", artist="")
    store.add_track(track)
    lines = [
        Line(text="hello world", start=1.0, progress=0.0,
             words=[Word(text="hello", start=1.0, end=1.5),
                    Word(text="world", start=1.5, end=2.0)]),
        Line(text="second line", start=3.0, progress=0.0,
             words=[Word(text="second", start=3.0, end=3.5),
                    Word(text="line", start=3.5, end=4.0)]),
    ]
    store.save_transcript(track.id, lines, backend="test")
    return track


def _cleanup_track(track_id):
    from app.pipeline import delete_track_files, store
    delete_track_files(track_id)
    store.delete_track(track_id)


def test_rename_track_title_and_artist():
    track = _make_editable_track()
    try:
        res = client.patch(f"/api/tracks/{track.id}",
                           json={"title": "Renamed", "artist": "Someone"})
        assert res.status_code == 200
        assert res.json() == {"id": track.id, "title": "Renamed",
                              "artist": "Someone"}
        # GET reflects it and exports use the new tags
        got = client.get(f"/api/tracks/{track.id}").json()
        assert got["title"] == "Renamed" and got["artist"] == "Someone"
        lrc = client.get(f"/api/tracks/{track.id}/export", params={"fmt": "lrc"}).text
        assert "[ti:Renamed]" in lrc and "[ar:Someone]" in lrc
    finally:
        _cleanup_track(track.id)


def test_rename_track_validates():
    track = _make_editable_track()
    try:
        assert client.patch(f"/api/tracks/{track.id}", json={}).status_code == 400
        assert client.patch(f"/api/tracks/{track.id}",
                            json={"title": "   "}).status_code == 400
        assert client.patch("/api/tracks/nope",
                            json={"title": "x"}).status_code == 404
        # artist can be cleared, title cannot
        assert client.patch(f"/api/tracks/{track.id}",
                            json={"artist": ""}).status_code == 200
        assert client.get(f"/api/tracks/{track.id}").json()["artist"] == ""
    finally:
        _cleanup_track(track.id)


def test_edit_line_keeps_surviving_word_timings():
    track = _make_editable_track()
    try:
        res = client.post(f"/api/tracks/{track.id}/edit_line",
                          json={"lineIndex": 0, "text": "hello word"})
        assert res.status_code == 200
        from app.pipeline import store
        lines = store.load_transcript(track.id)
        w = lines[0].words
        assert [x.text for x in w] == ["hello", "word"]
        assert w[0].start == 1.0 and w[1].start == 1.5   # timings preserved
        assert lines[1].words[0].start == 3.0            # next line untouched
    finally:
        _cleanup_track(track.id)


def test_edit_line_inserted_word_borrows_next_start():
    track = _make_editable_track()
    try:
        res = client.post(f"/api/tracks/{track.id}/edit_line",
                          json={"lineIndex": 0, "text": "hello big world"})
        assert res.status_code == 200
        from app.pipeline import store
        lines = store.load_transcript(track.id)
        w = lines[0].words
        assert [x.text for x in w] == ["hello", "big", "world"]
        assert w[1].start == 1.5          # zero-width, before the next tag
        assert w[2].start == 1.5          # replaced slot keeps its timing
        assert w[2].end == 2.0            # explicit end tag survives the round trip
    finally:
        _cleanup_track(track.id)


def test_edit_line_deletion_and_replacement_expansion():
    track = _make_editable_track()
    try:
        # delete a word
        res = client.post(f"/api/tracks/{track.id}/edit_line",
                          json={"lineIndex": 0, "text": "hello"})
        assert res.status_code == 200
        from app.pipeline import store
        lines = store.load_transcript(track.id)
        assert lines[0].text == "hello"
        assert lines[0].words[0].start == 1.0

        # one word replaced by two: both share the old slot's timing
        res = client.post(f"/api/tracks/{track.id}/edit_line",
                          json={"lineIndex": 1, "text": "second long line"})
        assert res.status_code == 200
        from app.pipeline import store
        lines = store.load_transcript(track.id)
        w = lines[1].words
        assert [x.text for x in w] == ["second", "long", "line"]
        assert w[0].start == 3.0 and w[1].start == 3.5 and w[2].start == 3.5
    finally:
        _cleanup_track(track.id)


def test_edit_line_validates():
    track = _make_editable_track()
    try:
        assert client.post(f"/api/tracks/{track.id}/edit_line",
                           json={"lineIndex": 9, "text": "x"}).status_code == 400
        assert client.post(f"/api/tracks/{track.id}/edit_line",
                           json={"lineIndex": 0, "text": "   "}).status_code == 400
        assert client.post("/api/tracks/nope/edit_line",
                           json={"lineIndex": 0, "text": "x"}).status_code == 404
        # the failed edits did NOT create versions
        idx = client.get(f"/api/tracks/{track.id}/versions").json()
        assert len(idx["versions"]) == 1
        # successful edit creates an undoable "edited" version
        client.post(f"/api/tracks/{track.id}/edit_line",
                    json={"lineIndex": 0, "text": "hello there"})
        idx = client.get(f"/api/tracks/{track.id}/versions").json()
        assert idx["active"] == 2
        assert idx["versions"][-1]["note"] == "edited"
        assert idx["versions"][-1]["backend"] is None
    finally:
        _cleanup_track(track.id)
