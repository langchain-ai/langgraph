from agent import MAX_TARGETS, _questions, _target_ids


def observation(tree: str, selectors: dict[str, str]):
    return {
        "url": "https://example.com",
        "title": "Example",
        "tree": tree,
        "fingerprint": "fingerprint",
        "selectors": selectors,
    }


def test_target_ids_only_include_snapshot_ids_with_selectors() -> None:
    page = observation(
        "[submit] button: Submit\n[query] textbox: Search\n[missing] link: Missing",
        {"submit": "/html/body/button", "query": "/html/body/input"},
    )

    assert _target_ids(page) == ["submit", "query"]


def test_target_ids_support_frame_scoped_ids() -> None:
    page = observation(
        "[0-12] button: Search\n  [2-7] textbox: Destination",
        {"0-12": "/html/body/button", "2-7": "/html/body/iframe/html/body/input"},
    )

    assert _target_ids(page) == ["0-12", "2-7"]


def test_target_ids_respect_typesafe_choice_limit() -> None:
    selectors = {str(index): f"/html/body/button[{index}]" for index in range(MAX_TARGETS + 1)}
    tree = "\n".join(f"[{index}] button: Option {index}" for index in range(MAX_TARGETS + 1))

    assert _target_ids(observation(tree, selectors)) == [str(index) for index in range(MAX_TARGETS)]


def test_questions_fan_out_operation_and_targets() -> None:
    page = observation(
        "[submit] button: Submit\n[query] textbox: Search",
        {"submit": "/html/body/button", "query": "/html/body/input"},
    )

    questions = _questions(page, "Search for LangGraph")

    assert set(questions) == {"operation", "click_target", "type_text_target"}
    assert "STOP_SIDE_EFFECT" in questions["operation"].criteria
    assert questions["click_target"].criteria == {"submit": None, "query": None}


def test_questions_hide_targeted_operations_without_targets() -> None:
    questions = _questions(observation("[root] document: Empty", {}), "Find a result")

    assert set(questions) == {"operation"}
    assert "CLICK" not in questions["operation"].criteria
    assert "TYPE_TEXT" not in questions["operation"].criteria
