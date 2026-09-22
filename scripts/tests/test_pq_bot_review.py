#!/usr/bin/env python3
"""Contract tests for the review trigger and comment-only publisher."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pq_bot_review as review


def comment(body, association="MEMBER", is_pr=True):
    return {
        "issue": {"number": 12, "pull_request": {} if is_pr else None},
        "comment": {"body": body, "author_association": association},
    }


class ReviewTriggerTests(unittest.TestCase):
    def test_slash_and_literal_mentions_work_without_machine_user(self):
        self.assertEqual((12, False), review.selected_review("issue_comment", comment("/pq-bot review"), ""))
        self.assertEqual((12, False), review.selected_review("issue_comment", comment("@pq-bot review"), ""))

    def test_machine_user_mention_and_review_request(self):
        self.assertEqual(
            (12, False),
            review.selected_review("issue_comment", comment("@molarverse-pq-bot review"), "molarverse-pq-bot"),
        )
        event = {
            "action": "review_requested",
            "requested_reviewer": {"login": "molarverse-pq-bot"},
            "pull_request": {"number": 23},
        }
        self.assertEqual((23, True), review.selected_review("pull_request_target", event, "molarverse-pq-bot"))
        self.assertIsNone(review.selected_review("pull_request_target", event, ""))

    def test_quotes_outsiders_and_other_commands_do_not_run(self):
        for body in (
            "Someone said @pq-bot review",
            "> @pq-bot review",
            "    /pq-bot review",
            "```\n/pq-bot review\n```",
            "/pq-bot fix #3",
        ):
            self.assertIsNone(review.selected_review("issue_comment", comment(body), ""))
        self.assertIsNone(review.selected_review("issue_comment", comment("/pq-bot review", "NONE"), ""))
        self.assertIsNone(review.selected_review("issue_comment", comment("/pq-bot review", is_pr=False), ""))


class ReviewOutputTests(unittest.TestCase):
    def test_added_lines_across_hunks(self):
        patch = "@@ -2,3 +2,4 @@\n old\n-removed\n+added\n+++literal pluses\n end\n@@ -20,1 +21,1 @@\n-later\n+replacement\n"
        self.assertEqual({3, 4, 21}, review.added_lines(patch))

    def test_only_changed_lines_are_published(self):
        result = {
            "summary": "Found one regression.",
            "findings": [
                {"path": "src/a.cpp", "line": 8, "body": "This path can dereference null."},
                {"path": "src/a.cpp", "line": 9, "body": "Not a changed line."},
                {"path": ".github/workflows/b.yml", "line": 8, "body": "Wrong file."},
            ],
        }
        summary, comments = review.validated_review(result, {"src/a.cpp": [8]})
        self.assertEqual("Found one regression.", summary)
        self.assertEqual([{"path": "src/a.cpp", "line": 8, "side": "RIGHT", "body": "This path can dereference null."}], comments)

    def test_reads_only_model_text_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "events.jsonl")
            path.write_text(
                json.dumps({"type": "tool_use", "part": {"text": "ignore"}}) + "\n"
                + json.dumps({"type": "text", "part": {"text": '{"summary":"Clear","findings":[]}'}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual("Clear", review.read_model_result(path)["summary"])

    def test_publisher_never_approves_and_removes_requested_alias(self):
        context = {
            "repo": "MolarVerse/PQ",
            "number": 12,
            "head_sha": "abc123",
            "base_sha": "base123",
            "allowed_lines": {"src/a.cpp": [8]},
            "machine_login": "molarverse-pq-bot",
            "requested": True,
        }
        events = {
            "type": "text",
            "part": {
                "text": json.dumps(
                    {
                        "summary": "One issue",
                        "findings": [{"path": "src/a.cpp", "line": 8, "body": "Fix this."}],
                    }
                )
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "pq-review-context.json").write_text(json.dumps(context), encoding="utf-8")
            Path(directory, "pq-review-events.jsonl").write_text(json.dumps(events) + "\n", encoding="utf-8")
            pull = {"state": "open", "head": {"sha": "abc123"}, "base": {"sha": "base123"}, "requested_reviewers": [{"login": "molarverse-pq-bot"}]}
            with mock.patch.dict("os.environ", {"GH_TOKEN": "test-token"}), mock.patch.object(review, "api", side_effect=[pull, {}, pull, {}]) as api:
                review.publish(directory)
            self.assertEqual("COMMENT", api.call_args_list[1].args[3]["event"])
            self.assertEqual("DELETE", api.call_args_list[3].args[0])

    def test_publisher_aborts_if_head_changed(self):
        context = {"repo": "MolarVerse/PQ", "number": 12, "head_sha": "abc123", "base_sha": "base123", "allowed_lines": {}, "requested": False}
        events = {"type": "text", "part": {"text": '{"summary":"No issue","findings":[]}'}}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "pq-review-context.json").write_text(json.dumps(context), encoding="utf-8")
            Path(directory, "pq-review-events.jsonl").write_text(json.dumps(events) + "\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"GH_TOKEN": "test-token"}), mock.patch.object(
                review, "api", return_value={"state": "open", "head": {"sha": "new"}, "base": {"sha": "base123"}}
            ) as api:
                with self.assertRaisesRegex(ValueError, "changed during review"):
                    review.publish(directory)
            api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
