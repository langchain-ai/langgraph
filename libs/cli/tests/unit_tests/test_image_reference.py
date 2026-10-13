import pytest

from langgraph_cli.image_reference import ImageReference


@pytest.mark.parametrize(
    ("reference", "repository", "tag"),
    [
        pytest.param(
            "registry.example.com/team/app:v1",
            "registry.example.com/team/app",
            "v1",
            id="tag_after_last_slash",
        ),
        pytest.param(
            "registry.example.com/team/app",
            "registry.example.com/team/app",
            None,
            id="no_tag",
        ),
        pytest.param(
            "localhost:5000/app",
            "localhost:5000/app",
            None,
            id="registry_port_is_not_a_tag",
        ),
        pytest.param(
            "localhost:5000/app:latest",
            "localhost:5000/app",
            "latest",
            id="registry_port_with_tag",
        ),
        pytest.param("app:dev", "app", "dev", id="bare_name_with_tag"),
    ],
)
def test_parse_splits_repository_and_tag(reference, repository, tag):
    assert ImageReference.parse(reference) == ImageReference(repository, tag)


def test_with_tag_replaces_the_tag():
    assert ImageReference("r/app", "v1").with_tag("v2") == ImageReference("r/app", "v2")


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        pytest.param(ImageReference("r/app", "v1"), "r/app:v1", id="tagged"),
        pytest.param(ImageReference("r/app"), "r/app", id="untagged"),
    ],
)
def test_str_renders_the_docker_reference(reference, expected):
    assert str(reference) == expected


@pytest.mark.parametrize(
    ("repo_digest", "expected"),
    [
        pytest.param("localhost:5000/app@sha256:abc", True, id="same_repository"),
        pytest.param("localhost:5000/app-2@sha256:abc", False, id="other_repository"),
        pytest.param("mirror.example.com/app@sha256:abc", False, id="other_registry"),
    ],
)
def test_matches_digest_only_for_the_same_repository(repo_digest, expected):
    assert ImageReference("localhost:5000/app", "v1").matches_digest(repo_digest) is (
        expected
    )


def test_parse_rejects_a_digest_reference():
    with pytest.raises(ValueError, match="digest"):
        ImageReference.parse("registry.example.com/app@sha256:abc")
