import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langsmith import trace, tracing_context

from gtm_agent import data_service

with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-12853"
        self.record = data_service.PROSPECTS[self.prospect_id]
        self.safe_profile = {
            "prospect_id": self.prospect_id,
            **{k: v for k, v in self.record.items() if k != "billing_qualification"},
        }
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled="local", client=Mock())
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive_value in self.record["billing_qualification"].values():
            self.assertNotIn(sensitive_value, serialized)

    def assert_safe_traces(self, run):
        for child in run.child_runs:
            self.assert_no_billing(child.inputs)
            self.assert_no_billing(child.outputs)
            self.assert_safe_traces(child)

    def test_lookup_returns_only_approved_fields(self):
        with patch.dict(self.record, {"unapproved_field": "internal"}):
            result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"found": True, "prospect": {
            "prospect_id": self.prospect_id,
            **{k: self.record[k] for k in (
                "name", "email", "annual_revenue", "disqualified", "enrichment_source"
            )},
        }})
        self.assert_no_billing(result)

    def test_new_profile_is_saved_and_traced_without_billing(self):
        with trace("build workflow") as run:
            result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            self.score_and_check(result["prospect_profile"])
        self.assertTrue(run.child_runs)
        self.assertEqual(result, {"found": True, "prospect_profile": self.safe_profile})
        self.assertEqual(data_service._PROFILES[self.prospect_id], self.safe_profile)
        self.assert_safe_traces(run)
        self.assertIn("billing_qualification", self.record)

    def test_cached_profile_and_cache_trace_preserve_nonsensitive_fields(self):
        cached = {"prospect_id": self.prospect_id, **self.record, "custom_field": "retained"}
        data_service._PROFILES[self.prospect_id] = cached
        expected = {**self.safe_profile, "custom_field": "retained"}
        runs = []
        retrieved = data_service.get_profile_from_db(
            self.prospect_id, langsmith_extra={"on_end": runs.append}
        )
        self.assertEqual(retrieved, {"prospect_profile": expected})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, retrieved)
        self.assert_no_billing(runs[0].outputs)
        with trace("cached workflow") as run:
            result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            self.score_and_check(result["prospect_profile"])
        self.assertEqual(result, {"found": True, "prospect_profile": expected})
        self.assert_safe_traces(run)

    def score_and_check(self, profile):
        offering = data_service.OFFERINGS["OFFER-10001"]
        score = {"score": 70, "justification": "fit"}
        model = Mock()
        model.invoke.return_value.model_dump.return_value = score
        with patch.object(agent, "_scoring_llm", model):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(result, score)
        messages = model.invoke.call_args.args[0]
        self.assert_no_billing(messages)
        offering_text, profile_text = messages[1]["content"].split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering_text.removeprefix("Offering:\n")), offering)
        expected = {k: v for k, v in profile.items() if k != "billing_qualification"}
        if "prospect_id" in expected:
            expected["tech_stack"] = self.record["tech_stack"]
        self.assertEqual(json.loads(profile_text), expected)

    def test_scoring_filters_legacy_profiles_before_serialization(self):
        for has_id in (True, False):
            with self.subTest(has_id=has_id):
                profile = {**self.record, "tech_stack": ["stale"], "custom_field": "retained"}
                if has_id:
                    profile["prospect_id"] = self.prospect_id
                self.score_and_check(profile)
                self.assertIn("billing_qualification", profile)

    def test_missing_profiles_and_prospects(self):
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
