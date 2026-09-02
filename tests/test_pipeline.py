import unittest

from face_occurrence_detector.pipeline import _build_target_embeddings


class FakeMatcher:
    def __init__(self):
        self.calls = []

    def build_target_embeddings(self, targets):
        self.calls.append(targets)
        return ["embedding"]


class BuildTargetEmbeddingsTests(unittest.TestCase):
    def test_discovery_only_run_does_not_require_reference_images(self):
        matcher = FakeMatcher()

        embeddings = _build_target_embeddings(matcher, [], discover_people=True)

        self.assertEqual(embeddings, [])
        self.assertEqual(matcher.calls, [])

    def test_targeted_run_delegates_to_matcher(self):
        matcher = FakeMatcher()

        embeddings = _build_target_embeddings(
            matcher,
            ["target.jpg"],
            discover_people=False,
        )

        self.assertEqual(embeddings, ["embedding"])
        self.assertEqual(matcher.calls, [["target.jpg"]])

    def test_empty_targeted_run_is_rejected(self):
        matcher = FakeMatcher()

        with self.assertRaisesRegex(
            ValueError,
            "enable person discovery",
        ):
            _build_target_embeddings(matcher, [], discover_people=False)


if __name__ == "__main__":
    unittest.main()
