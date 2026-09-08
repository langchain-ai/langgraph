from langgraph.pregel.remote import RemoteException


def test_dict_payload_is_preserved_and_rendered():
    payload = {
        "error": "InvalidUpdateError",
        "message": "bad update for key 'messages'",
    }
    exc = RemoteException(payload)
    assert exc.payload == payload
    msg = str(exc)
    assert "InvalidUpdateError" in msg
    assert "messages" in msg


def test_traceback_is_included():
    exc = RemoteException({"error": "boom", "traceback": "Traceback...\nline 1"})
    assert "Traceback..." in str(exc)


def test_string_payload_unchanged():
    exc = RemoteException("boom")
    assert str(exc) == "boom"
    assert exc.payload == "boom"


def test_non_dict_payload_falls_back_to_str():
    assert str(RemoteException(None)) == "None"
