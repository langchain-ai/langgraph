from unittest.mock import patch

from langgraph_cli.util import clean_empty_lines, warn_non_wolfi_distro


def test_clean_empty_lines():
    """Test clean_empty_lines function."""
    # Test with empty lines
    input_str = "line1\n\nline2\n\nline3"
    result = clean_empty_lines(input_str)
    assert result == "line1\nline2\nline3"

    # Test with no empty lines
    input_str = "line1\nline2\nline3"
    result = clean_empty_lines(input_str)
    assert result == "line1\nline2\nline3"

    # Test with only empty lines
    input_str = "\n\n\n"
    result = clean_empty_lines(input_str)
    assert result == ""

    # Test empty string
    input_str = ""
    result = clean_empty_lines(input_str)
    assert result == ""


def test_warn_non_wolfi_distro_with_debian(capsys):
    """Test that warning is shown when image_distro is 'debian'."""
    config = {"image_distro": "debian"}

    warn_non_wolfi_distro(config)

    captured = capsys.readouterr()
    assert (
        "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security."
        in captured.out
    )
    assert (
        "Wolfi is a security-oriented, minimal Linux distribution designed for containers."
        in captured.out
    )
    assert (
        'To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.'
        in captured.out
    )


def test_warn_non_wolfi_distro_with_default_debian(capsys):
    """Test that warning is shown when image_distro is missing (defaults to debian)."""
    config = {}  # No image_distro key, should default to debian

    warn_non_wolfi_distro(config)

    captured = capsys.readouterr()
    assert (
        "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security."
        in captured.out
    )
    assert (
        "Wolfi is a security-oriented, minimal Linux distribution designed for containers."
        in captured.out
    )
    assert (
        'To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.'
        in captured.out
    )


def test_warn_non_wolfi_distro_with_wolfi(capsys):
    """Test that no warning is shown when image_distro is 'wolfi'."""
    config = {"image_distro": "wolfi"}

    warn_non_wolfi_distro(config)

    captured = capsys.readouterr()
    assert captured.out == ""  # No output should be generated


def test_warn_non_wolfi_distro_with_other_distro(capsys):
    """Test that warning is shown when image_distro is something other than 'wolfi'."""
    config = {"image_distro": "ubuntu"}

    warn_non_wolfi_distro(config)

    captured = capsys.readouterr()
    assert (
        "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security."
        in captured.out
    )
    assert (
        "Wolfi is a security-oriented, minimal Linux distribution designed for containers."
        in captured.out
    )
    assert (
        'To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.'
        in captured.out
    )


def test_warn_non_wolfi_distro_output_formatting():
    """Test that the warning output is properly formatted with colors and empty line."""
    config = {"image_distro": "debian"}

    with patch("click.secho") as mock_secho:
        warn_non_wolfi_distro(config)

    # Verify click.secho was called with the correct parameters
    expected_calls = [
        (
            (
                "⚠️  Security Recommendation: Consider switching to Wolfi Linux for enhanced security.",
            ),
            {"fg": "yellow", "bold": True},
        ),
        (
            (
                "   Wolfi is a security-oriented, minimal Linux distribution designed for containers.",
            ),
            {"fg": "yellow"},
        ),
        (
            (
                '   To switch, add \'"image_distro": "wolfi"\' to your langgraph.json config file.',
            ),
            {"fg": "yellow"},
        ),
        (
            ("",),  # Empty line
            {},
        ),
    ]

    assert mock_secho.call_count == 4
    for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
        actual_call = mock_secho.call_args_list[i]
        assert actual_call.args == expected_args
        assert actual_call.kwargs == expected_kwargs


def test_warn_non_wolfi_distro_various_configs(capsys):
    """Test warn_non_wolfi_distro with various config scenarios."""
    test_cases = [
        # (config, should_warn, description)
        ({"image_distro": "debian"}, True, "explicit debian"),
        ({"image_distro": "wolfi"}, False, "explicit wolfi"),
        ({}, True, "missing image_distro (defaults to debian)"),
        ({"image_distro": "alpine"}, True, "other distro"),
        ({"image_distro": "ubuntu"}, True, "ubuntu distro"),
        ({"other_config": "value"}, True, "unrelated config keys"),
    ]

    for config, should_warn, description in test_cases:
        # Clear any previous output
        capsys.readouterr()

        warn_non_wolfi_distro(config)

        captured = capsys.readouterr()
        if should_warn:
            assert "⚠️  Security Recommendation" in captured.out, (
                f"Should warn for {description}"
            )
            assert "Wolfi" in captured.out, f"Should mention Wolfi for {description}"
        else:
            assert captured.out == "", f"Should not warn for {description}"


def test_warn_non_wolfi_distro_return_value():
    """Test that warn_non_wolfi_distro returns None."""
    config = {"image_distro": "debian"}
    result = warn_non_wolfi_distro(config)
    assert result is None

    config = {"image_distro": "wolfi"}
    result = warn_non_wolfi_distro(config)
    assert result is None


def test_warn_non_wolfi_distro_does_not_modify_config():
    """Test that warn_non_wolfi_distro does not modify the input config."""
    original_config = {"image_distro": "debian", "other_key": "value"}
    config_copy = original_config.copy()

    warn_non_wolfi_distro(config_copy)

    assert config_copy == original_config  # Config should remain unchanged
