from langgraph_cli.deploy import format_log_entry, format_timestamp, level_fg


class TestFormatTimestamp:
    def test_epoch_ms(self):
        assert format_timestamp(1773119644012) == "2026-03-10 05:14:04"

    def test_string_passthrough(self):
        assert format_timestamp("2026-03-08T00:00:00Z") == "2026-03-08T00:00:00Z"

    def test_empty(self):
        assert format_timestamp("") == ""

    def test_none(self):
        assert format_timestamp(None) == ""


class TestFormatLogEntry:
    def test_full_entry_epoch(self):
        entry = {"timestamp": 1773119644012, "level": "ERROR", "message": "boom"}
        result = format_log_entry(entry)
        assert result == "[2026-03-10 05:14:04] [ERROR] boom"

    def test_full_entry_string(self):
        entry = {
            "timestamp": "2026-03-08T12:00:00Z",
            "level": "ERROR",
            "message": "boom",
        }
        assert format_log_entry(entry) == "[2026-03-08T12:00:00Z] [ERROR] boom"

    def test_no_level(self):
        entry = {"timestamp": "2026-03-08T12:00:00Z", "message": "hello"}
        assert format_log_entry(entry) == "[2026-03-08T12:00:00Z] hello"

    def test_no_timestamp(self):
        entry = {"message": "bare message"}
        assert format_log_entry(entry) == "bare message"

    def test_empty_entry(self):
        assert format_log_entry({}) == ""


class TestLevelFg:
    def test_error(self):
        assert level_fg("ERROR") == "red"

    def test_error_lowercase(self):
        assert level_fg("error") == "red"

    def test_warning(self):
        assert level_fg("WARNING") == "yellow"

    def test_info_returns_none(self):
        assert level_fg("INFO") is None

    def test_empty_returns_none(self):
        assert level_fg("") is None
